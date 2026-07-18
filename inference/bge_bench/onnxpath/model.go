// Package onnxpath runs BGE-small-en-v1.5 inference via ONNX Runtime,
// mirroring Riffle's existing embedder pattern but with CLS-token pooling
// instead of mean pooling — BGE's documented sentence-embedding method.
package onnxpath

import (
	"fmt"
	"math"
	"sync"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/allank/onnx_test/bge_bench/tokenizer"
)

const hiddenSize = 384

var initOnce sync.Once
var initErr error

// Embedder wraps an ONNX Runtime session for BGE-small.
type Embedder struct {
	session *ort.DynamicAdvancedSession
	tok     *tokenizer.Tokenizer
}

// New loads the ONNX model and tokenizer, initializing the ONNX Runtime
// shared library at libPath (e.g. /usr/local/lib/libonnxruntime.dylib).
func New(modelPath, tokenizerPath, libPath string) (*Embedder, error) {
	initOnce.Do(func() {
		ort.SetSharedLibraryPath(libPath)
		initErr = ort.InitializeEnvironment()
	})
	if initErr != nil {
		return nil, fmt.Errorf("onnxruntime init: %w", initErr)
	}
	tok, err := tokenizer.LoadFile(tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}
	inputNames := []string{"input_ids", "attention_mask", "token_type_ids"}
	outputNames := []string{"last_hidden_state"}
	session, err := ort.NewDynamicAdvancedSession(modelPath, inputNames, outputNames, nil)
	if err != nil {
		return nil, fmt.Errorf("onnx session: %w", err)
	}
	return &Embedder{session: session, tok: tok}, nil
}

// Embed tokenizes text and returns the L2-normalized [CLS] sentence
// embedding.
func (e *Embedder) Embed(text string) ([]float32, error) {
	enc := e.tok.Encode(text, 512)
	seqLen := int64(len(enc.InputIDs))

	inputIDs, err := ort.NewTensor(ort.NewShape(1, seqLen), enc.InputIDs)
	if err != nil {
		return nil, err
	}
	defer inputIDs.Destroy()

	attnMask, err := ort.NewTensor(ort.NewShape(1, seqLen), enc.AttentionMask)
	if err != nil {
		return nil, err
	}
	defer attnMask.Destroy()

	tokenTypes, err := ort.NewTensor(ort.NewShape(1, seqLen), enc.TokenTypeIDs)
	if err != nil {
		return nil, err
	}
	defer tokenTypes.Destroy()

	outputData := make([]float32, seqLen*hiddenSize)
	output, err := ort.NewTensor(ort.NewShape(1, seqLen, hiddenSize), outputData)
	if err != nil {
		return nil, err
	}
	defer output.Destroy()

	if err := e.session.Run([]ort.Value{inputIDs, attnMask, tokenTypes}, []ort.Value{output}); err != nil {
		return nil, fmt.Errorf("onnx run: %w", err)
	}

	hidden := output.GetData()
	return clsAndNormalize(hidden), nil
}

func (e *Embedder) Close() error {
	return e.session.Destroy()
}

// clsAndNormalize takes the first token's ([CLS]) hidden state and
// L2-normalizes it. BGE's model card is explicit that this is the
// correct pooling method, not the BERT pooler head.
func clsAndNormalize(hidden []float32) []float32 {
	cls := make([]float32, hiddenSize)
	copy(cls, hidden[:hiddenSize])
	var norm float64
	for _, v := range cls {
		norm += float64(v) * float64(v)
	}
	norm = math.Sqrt(norm)
	if norm > 0 {
		for i := range cls {
			cls[i] = float32(float64(cls[i]) / norm)
		}
	}
	return cls
}
