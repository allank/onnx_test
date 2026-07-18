// Package puregopath is a hand-written, CGo-free forward pass for the
// BERT-family architecture used by BAAI/bge-small-en-v1.5: 12 layers,
// hidden size 384, 12 attention heads, intermediate size 1536, exact
// (erf-based) GELU, post-LN residual blocks. Weights are loaded directly
// from the model's safetensors file. Sentence embedding is the [CLS]
// token (position 0) of the final hidden state, L2-normalized — BGE's
// documented pooling method (not the BERT pooler head).
package puregopath

import (
	"fmt"
	"math"

	"github.com/allank/onnx_test/bge_bench/safetensors"
	"gonum.org/v1/gonum/mat"
)

const (
	hiddenSize       = 384
	numLayers        = 12
	numHeads         = 12
	headDim          = hiddenSize / numHeads
	intermediateSize = 1536
	layerNormEps     = 1e-12
)

// mat64 is a row-major dense matrix backed by a flat slice. Rows are
// contiguous, so linear() can dot two rows directly without an
// intervening transpose or bounds-checked accessor call.
type mat64 struct {
	data       []float64
	rows, cols int
}

func newMat64(rows, cols int) mat64 {
	return mat64{data: make([]float64, rows*cols), rows: rows, cols: cols}
}

func (m mat64) row(i int) []float64 {
	return m.data[i*m.cols : (i+1)*m.cols]
}

type layerWeights struct {
	qW, kW, vW               mat64
	qB, kB, vB               []float64
	attnOutW                 mat64
	attnOutB                 []float64
	attnLNWeight, attnLNBias []float64
	interW                   mat64
	interB                   []float64
	outW                     mat64
	outB                     []float64
	outLNWeight, outLNBias   []float64
}

// Model holds BGE-small's weights loaded into row-major matrices.
type Model struct {
	wordEmb, posEmb, typeEmb mat64
	embLNWeight, embLNBias   []float64
	layers                   [numLayers]layerWeights
}

type loader struct {
	f   *safetensors.File
	err error
}

// dense loads a tensor with PyTorch nn.Linear's [out_features, in_features]
// row-major layout, upconverting to float64 for arithmetic.
func (l *loader) dense(name string, rows, cols int) mat64 {
	if l.err != nil {
		return mat64{}
	}
	data, err := l.f.F32(name)
	if err != nil {
		l.err = err
		return mat64{}
	}
	if len(data) != rows*cols {
		l.err = fmt.Errorf("puregopath: tensor %q has %d elements, want %d (%dx%d)", name, len(data), rows*cols, rows, cols)
		return mat64{}
	}
	d64 := make([]float64, len(data))
	for i, v := range data {
		d64[i] = float64(v)
	}
	return mat64{data: d64, rows: rows, cols: cols}
}

func (l *loader) vec(name string, n int) []float64 {
	if l.err != nil {
		return nil
	}
	data, err := l.f.F32(name)
	if err != nil {
		l.err = err
		return nil
	}
	if len(data) != n {
		l.err = fmt.Errorf("puregopath: tensor %q has %d elements, want %d", name, len(data), n)
		return nil
	}
	out := make([]float64, n)
	for i, v := range data {
		out[i] = float64(v)
	}
	return out
}

// Load reads a BGE-small-en-v1.5 model.safetensors file and prepares it
// for pure-Go inference.
func Load(path string) (*Model, error) {
	f, err := safetensors.Load(path)
	if err != nil {
		return nil, err
	}
	l := &loader{f: f}
	m := &Model{}

	m.wordEmb = l.dense("embeddings.word_embeddings.weight", 30522, hiddenSize)
	m.posEmb = l.dense("embeddings.position_embeddings.weight", 512, hiddenSize)
	m.typeEmb = l.dense("embeddings.token_type_embeddings.weight", 2, hiddenSize)
	m.embLNWeight = l.vec("embeddings.LayerNorm.weight", hiddenSize)
	m.embLNBias = l.vec("embeddings.LayerNorm.bias", hiddenSize)

	for i := 0; i < numLayers; i++ {
		p := fmt.Sprintf("encoder.layer.%d.", i)
		lw := &m.layers[i]
		lw.qW = l.dense(p+"attention.self.query.weight", hiddenSize, hiddenSize)
		lw.qB = l.vec(p+"attention.self.query.bias", hiddenSize)
		lw.kW = l.dense(p+"attention.self.key.weight", hiddenSize, hiddenSize)
		lw.kB = l.vec(p+"attention.self.key.bias", hiddenSize)
		lw.vW = l.dense(p+"attention.self.value.weight", hiddenSize, hiddenSize)
		lw.vB = l.vec(p+"attention.self.value.bias", hiddenSize)
		lw.attnOutW = l.dense(p+"attention.output.dense.weight", hiddenSize, hiddenSize)
		lw.attnOutB = l.vec(p+"attention.output.dense.bias", hiddenSize)
		lw.attnLNWeight = l.vec(p+"attention.output.LayerNorm.weight", hiddenSize)
		lw.attnLNBias = l.vec(p+"attention.output.LayerNorm.bias", hiddenSize)
		lw.interW = l.dense(p+"intermediate.dense.weight", intermediateSize, hiddenSize)
		lw.interB = l.vec(p+"intermediate.dense.bias", intermediateSize)
		lw.outW = l.dense(p+"output.dense.weight", hiddenSize, intermediateSize)
		lw.outB = l.vec(p+"output.dense.bias", hiddenSize)
		lw.outLNWeight = l.vec(p+"output.LayerNorm.weight", hiddenSize)
		lw.outLNBias = l.vec(p+"output.LayerNorm.bias", hiddenSize)
	}
	if l.err != nil {
		return nil, l.err
	}
	return m, nil
}

// Embed runs the full forward pass for one (unpadded, unbatched) sequence
// of token IDs and returns the L2-normalized [CLS] sentence embedding.
func (m *Model) Embed(inputIDs []int64) []float64 {
	seqLen := len(inputIDs)
	x := newMat64(seqLen, hiddenSize)
	for i, id := range inputIDs {
		xi := x.row(i)
		we := m.wordEmb.row(int(id))
		pe := m.posEmb.row(i)
		te := m.typeEmb.row(0)
		for d := 0; d < hiddenSize; d++ {
			xi[d] = we[d] + pe[d] + te[d]
		}
	}
	layerNormInPlace(x, m.embLNWeight, m.embLNBias)

	for l := 0; l < numLayers; l++ {
		x = attentionBlock(x, &m.layers[l])
		x = ffnBlock(x, &m.layers[l])
	}

	cls := make([]float64, hiddenSize)
	copy(cls, x.row(0))
	normalize(cls)
	return cls
}

func attentionBlock(x mat64, lw *layerWeights) mat64 {
	seqLen := x.rows
	q := linear(x, lw.qW, lw.qB)
	k := linear(x, lw.kW, lw.kB)
	v := linear(x, lw.vW, lw.vB)

	context := newMat64(seqLen, hiddenSize)
	scores := newMat64(seqLen, seqLen)
	scale := 1.0 / math.Sqrt(float64(headDim))

	for h := 0; h < numHeads; h++ {
		off := h * headDim

		for i := 0; i < seqLen; i++ {
			qi := q.row(i)[off : off+headDim]
			srow := scores.row(i)
			for j := 0; j < seqLen; j++ {
				kj := k.row(j)[off : off+headDim]
				var dot float64
				for d := 0; d < headDim; d++ {
					dot += qi[d] * kj[d]
				}
				srow[j] = dot * scale
			}
		}

		softmaxRows(scores)

		for i := 0; i < seqLen; i++ {
			srow := scores.row(i)
			crow := context.row(i)[off : off+headDim]
			for d := range crow {
				crow[d] = 0
			}
			for j := 0; j < seqLen; j++ {
				s := srow[j]
				vj := v.row(j)[off : off+headDim]
				for d := 0; d < headDim; d++ {
					crow[d] += s * vj[d]
				}
			}
		}
	}

	attnOut := linear(context, lw.attnOutW, lw.attnOutB)
	addInPlace(attnOut, x)
	layerNormInPlace(attnOut, lw.attnLNWeight, lw.attnLNBias)
	return attnOut
}

func ffnBlock(x mat64, lw *layerWeights) mat64 {
	inter := linear(x, lw.interW, lw.interB)
	geluInPlace(inter)
	out := linear(inter, lw.outW, lw.outB)
	addInPlace(out, x)
	layerNormInPlace(out, lw.outLNWeight, lw.outLNBias)
	return out
}

// linear computes x @ w^T + b, where w follows the PyTorch Linear
// convention of shape [out_features, in_features]. The matmul itself is
// delegated to gonum, whose Dgemm kernel is hand-written per-arch assembly
// (SIMD dot products, cache blocking) — a naive Go triple loop over raw
// slices is measurably *slower* here despite avoiding bounds-checked
// accessors, since the Go compiler doesn't vectorize float64 FMA loops the
// way gonum's asm does. Only the bias add (a cheap elementwise pass) is
// done via raw slice access, since that part gonum doesn't fuse in for us.
func linear(x, w mat64, b []float64) mat64 {
	seqLen, outDim := x.rows, w.rows
	xD := mat.NewDense(x.rows, x.cols, x.data)
	wD := mat.NewDense(w.rows, w.cols, w.data)
	yD := mat.NewDense(seqLen, outDim, nil)
	yD.Mul(xD, wD.T())

	y := mat64{data: yD.RawMatrix().Data, rows: seqLen, cols: outDim}
	for i := 0; i < seqLen; i++ {
		yi := y.row(i)
		for j, bj := range b {
			yi[j] += bj
		}
	}
	return y
}

func layerNormInPlace(x mat64, weight, bias []float64) {
	for i := 0; i < x.rows; i++ {
		row := x.row(i)
		var mean float64
		for _, v := range row {
			mean += v
		}
		mean /= float64(len(row))
		var variance float64
		for _, v := range row {
			d := v - mean
			variance += d * d
		}
		variance /= float64(len(row))
		denom := math.Sqrt(variance + layerNormEps)
		for j, v := range row {
			row[j] = (v-mean)/denom*weight[j] + bias[j]
		}
	}
}

func softmaxRows(x mat64) {
	for i := 0; i < x.rows; i++ {
		row := x.row(i)
		maxV := math.Inf(-1)
		for _, v := range row {
			if v > maxV {
				maxV = v
			}
		}
		var sum float64
		for j, v := range row {
			e := math.Exp(v - maxV)
			row[j] = e
			sum += e
		}
		invSum := 1 / sum
		for j := range row {
			row[j] *= invSum
		}
	}
}

func geluInPlace(x mat64) {
	for i, v := range x.data {
		x.data[i] = 0.5 * v * (1 + math.Erf(v/math.Sqrt2))
	}
}

func addInPlace(dst, other mat64) {
	for i := range dst.data {
		dst.data[i] += other.data[i]
	}
}

func normalize(v []float64) {
	var sumSq float64
	for _, x := range v {
		sumSq += x * x
	}
	norm := math.Sqrt(sumSq)
	if norm == 0 {
		return
	}
	for i := range v {
		v[i] /= norm
	}
}
