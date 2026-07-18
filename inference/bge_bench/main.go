// Command bge_bench benchmarks pure-Go BERT inference (puregopath) against
// ONNX Runtime (onnxpath) for BAAI/bge-small-en-v1.5, on identical inputs,
// reporting both correctness (cosine similarity between the two paths'
// output vectors) and throughput/latency.
package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"runtime/pprof"
	"time"

	"github.com/allank/onnx_test/bge_bench/onnxpath"
	"github.com/allank/onnx_test/bge_bench/puregopath"
	"github.com/allank/onnx_test/bge_bench/tokenizer"
)

var testSentences = []string{
	"OAuth token refresh",
	"Describes the refresh token rotation pattern used in single page applications.",
	"security/oauth2\ntoken-refresh.md pkce-flow.md client-credentials.md\nOAuth 2.0 Token Refresh\nDescribes the refresh token rotation pattern used in SPAs\nRelated: [[PKCE Flow]], [[Token Storage]]",
	"Riffle is a single-binary CLI tool written in Go that builds and queries a semantic index of a local directory hierarchy. It allows users, and especially LLM agents, to find conceptually relevant folders within a large file tree quickly, without scanning individual files. A query like OAuth token refresh returns the three most relevant folder paths in milliseconds, consuming fewer than fifty tokens when passed to an LLM. The primary target use case is an Obsidian vault, a large folder of Markdown notes organised into a hierarchy of topics.",
	"The index combines two complementary strategies: structural integrity via Merkle hashing, and semantic search via vector embeddings.",
	"On re-index, Riffle compares the stored root hash to the recomputed root hash. If equal, nothing has changed. If different, it descends only into directories whose hash has changed, re-embedding only those.",
	"meeting notes",
	"What changed and why: the v0.4 design indexed at folder granularity and used a single HNSW vector store as its retrieval mechanism. That was a reasonable first cut, but the design problem has since been reframed more precisely as token optimisation rather than pure retrieval accuracy.",
}

func cosineSimilarity(a, b []float64) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

func toF64(v []float32) []float64 {
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = float64(x)
	}
	return out
}

func main() {
	dataDir := flag.String("data", "data", "directory containing model.onnx, model.safetensors, tokenizer.json")
	libPath := flag.String("onnxlib", "/usr/local/lib/libonnxruntime.dylib", "path to libonnxruntime shared library")
	iterations := flag.Int("iterations", 20, "number of timed passes over the test sentence set")
	cpuprofile := flag.String("cpuprofile", "", "if set, write a pprof CPU profile of the pure-Go timing loop to this path")
	flag.Parse()

	tokPath := *dataDir + "/tokenizer.json"
	onnxModelPath := *dataDir + "/model.onnx"
	safetensorsPath := *dataDir + "/model.safetensors"

	fmt.Println("Loading ONNX embedder...")
	onnxEmb, err := onnxpath.New(onnxModelPath, tokPath, *libPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "onnx load error:", err)
		os.Exit(1)
	}
	defer onnxEmb.Close()

	fmt.Println("Loading pure-Go model (parsing safetensors)...")
	loadStart := time.Now()
	pureModel, err := puregopath.Load(safetensorsPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "puregopath load error:", err)
		os.Exit(1)
	}
	fmt.Printf("Pure-Go model loaded in %s\n\n", time.Since(loadStart))

	tok, err := tokenizer.LoadFile(tokPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "tokenizer load error:", err)
		os.Exit(1)
	}

	// --- Correctness check: cosine similarity between the two paths ---
	fmt.Println("=== Correctness (cosine similarity, ONNX vs pure-Go) ===")
	minCos := 1.0
	for _, s := range testSentences {
		onnxVec, err := onnxEmb.Embed(s)
		if err != nil {
			fmt.Fprintln(os.Stderr, "onnx embed error:", err)
			os.Exit(1)
		}
		enc := tok.Encode(s, 512)
		pureVec := pureModel.Embed(enc.InputIDs)

		cos := cosineSimilarity(toF64(onnxVec), pureVec)
		if cos < minCos {
			minCos = cos
		}
		trunc := s
		if len(trunc) > 40 {
			trunc = trunc[:40] + "..."
		}
		fmt.Printf("  cos=%.6f  len=%3d  %q\n", cos, len(enc.InputIDs), trunc)
	}
	fmt.Printf("Minimum cosine similarity across test set: %.6f\n\n", minCos)

	// --- Timing ---
	fmt.Printf("=== Timing (%d iterations over %d sentences) ===\n", *iterations, len(testSentences))

	// Pre-tokenize once so tokenization cost isn't attributed to either
	// inference path.
	encodings := make([]tokenizer.Output, len(testSentences))
	for i, s := range testSentences {
		encodings[i] = tok.Encode(s, 512)
	}

	// ONNX timing
	onnxStart := time.Now()
	for i := 0; i < *iterations; i++ {
		for _, s := range testSentences {
			if _, err := onnxEmb.Embed(s); err != nil {
				fmt.Fprintln(os.Stderr, "onnx embed error:", err)
				os.Exit(1)
			}
		}
	}
	onnxElapsed := time.Since(onnxStart)
	onnxCount := *iterations * len(testSentences)

	// Pure-Go timing
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			fmt.Fprintln(os.Stderr, "cpuprofile create error:", err)
			os.Exit(1)
		}
		defer f.Close()
		if err := pprof.StartCPUProfile(f); err != nil {
			fmt.Fprintln(os.Stderr, "cpuprofile start error:", err)
			os.Exit(1)
		}
		defer pprof.StopCPUProfile()
	}
	pureStart := time.Now()
	for i := 0; i < *iterations; i++ {
		for _, enc := range encodings {
			_ = pureModel.Embed(enc.InputIDs)
		}
	}
	pureElapsed := time.Since(pureStart)
	pureCount := *iterations * len(encodings)

	report("ONNX Runtime", onnxElapsed, onnxCount)
	report("Pure-Go", pureElapsed, pureCount)

	fmt.Printf("\nPure-Go is %.1fx the per-embedding latency of ONNX Runtime.\n",
		(pureElapsed.Seconds()/float64(pureCount))/(onnxElapsed.Seconds()/float64(onnxCount)))
}

func report(label string, elapsed time.Duration, count int) {
	perEmbed := elapsed / time.Duration(count)
	throughput := float64(count) / elapsed.Seconds()
	fmt.Printf("  %-14s total=%-12s per-embedding=%-12s throughput=%.1f/sec\n",
		label, elapsed.Round(time.Millisecond), perEmbed.Round(time.Microsecond), throughput)
}
