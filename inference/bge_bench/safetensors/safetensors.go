// Package safetensors is a minimal, read-only parser for the safetensors
// format: an 8-byte little-endian header length, a JSON header describing
// each tensor's dtype/shape/byte offsets, followed by the raw tensor bytes.
// https://github.com/huggingface/safetensors#format
package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
)

type tensorMeta struct {
	DType       string `json:"dtype"`
	Shape       []int  `json:"shape"`
	DataOffsets [2]int `json:"data_offsets"`
}

// File holds a memory-mapped-in-full safetensors file and its parsed header.
type File struct {
	data    []byte // raw tensor data section
	tensors map[string]tensorMeta
}

// Load reads and parses a safetensors file fully into memory.
func Load(path string) (*File, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(raw) < 8 {
		return nil, fmt.Errorf("safetensors: file too short")
	}
	headerLen := binary.LittleEndian.Uint64(raw[:8])
	if uint64(len(raw)) < 8+headerLen {
		return nil, fmt.Errorf("safetensors: truncated header")
	}
	var rawHeader map[string]json.RawMessage
	if err := json.Unmarshal(raw[8:8+headerLen], &rawHeader); err != nil {
		return nil, fmt.Errorf("safetensors: parse header: %w", err)
	}
	tensors := make(map[string]tensorMeta, len(rawHeader))
	for name, msg := range rawHeader {
		if name == "__metadata__" {
			continue
		}
		var m tensorMeta
		if err := json.Unmarshal(msg, &m); err != nil {
			return nil, fmt.Errorf("safetensors: parse tensor %q: %w", name, err)
		}
		tensors[name] = m
	}
	return &File{data: raw[8+headerLen:], tensors: tensors}, nil
}

// Shape returns the shape of a named tensor.
func (f *File) Shape(name string) ([]int, error) {
	m, ok := f.tensors[name]
	if !ok {
		return nil, fmt.Errorf("safetensors: tensor %q not found", name)
	}
	return m.Shape, nil
}

// F32 returns a named tensor's data as float32, converting from its stored
// dtype if necessary. Only F32 is supported for now (all BGE-small weights
// are F32).
func (f *File) F32(name string) ([]float32, error) {
	m, ok := f.tensors[name]
	if !ok {
		return nil, fmt.Errorf("safetensors: tensor %q not found", name)
	}
	if m.DType != "F32" {
		return nil, fmt.Errorf("safetensors: tensor %q has unsupported dtype %s", name, m.DType)
	}
	start, end := m.DataOffsets[0], m.DataOffsets[1]
	if start < 0 || end > len(f.data) || start > end {
		return nil, fmt.Errorf("safetensors: tensor %q has invalid offsets [%d,%d)", name, start, end)
	}
	raw := f.data[start:end]
	n := len(raw) / 4
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		bits := binary.LittleEndian.Uint32(raw[i*4 : i*4+4])
		out[i] = math.Float32frombits(bits)
	}
	return out, nil
}

// Names returns all tensor names in the file.
func (f *File) Names() []string {
	names := make([]string, 0, len(f.tensors))
	for n := range f.tensors {
		names = append(names, n)
	}
	return names
}
