package puregopath

import (
	"reflect"
	"testing"
)

func TestConcatFloat64(t *testing.T) {
	cases := []struct {
		name  string
		parts [][]float64
		want  []float64
	}{
		{
			name:  "three non-empty slices",
			parts: [][]float64{{1, 2}, {3, 4, 5}, {6}},
			want:  []float64{1, 2, 3, 4, 5, 6},
		},
		{
			name:  "single slice",
			parts: [][]float64{{1, 2, 3}},
			want:  []float64{1, 2, 3},
		},
		{
			name:  "empty slice in the middle is skipped over",
			parts: [][]float64{{1}, {}, {2}},
			want:  []float64{1, 2},
		},
		{
			name:  "no slices",
			parts: [][]float64{},
			want:  []float64{},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := concatFloat64(tc.parts...)
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("concatFloat64(%v) = %v, want %v", tc.parts, got, tc.want)
			}
		})
	}
}

func TestFusedQKVLayout(t *testing.T) {
	// qW, kW, vW each 2x3 (rows=out_features, cols=in_features); fusing
	// them should stack rows in Q,K,V order with columns untouched, since
	// linear() expects w's rows to already be the per-output-feature
	// vectors to dot against x's rows.
	qW := mat64{data: []float64{1, 2, 3, 4, 5, 6}, rows: 2, cols: 3}
	kW := mat64{data: []float64{7, 8, 9, 10, 11, 12}, rows: 2, cols: 3}
	vW := mat64{data: []float64{13, 14, 15, 16, 17, 18}, rows: 2, cols: 3}

	fused := fuseWeights(qW, kW, vW)

	wantData := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}
	if !reflect.DeepEqual(fused.data, wantData) {
		t.Errorf("fused.data = %v, want %v", fused.data, wantData)
	}
	if fused.rows != 6 || fused.cols != 3 {
		t.Errorf("fused dims = %dx%d, want 6x3", fused.rows, fused.cols)
	}
}

func TestFuseWeightsPanicsOnColumnMismatch(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("fuseWeights did not panic on mismatched column counts")
		}
	}()
	qW := mat64{data: []float64{1, 2, 3, 4}, rows: 2, cols: 2}
	kW := mat64{data: []float64{1, 2, 3}, rows: 1, cols: 3}
	fuseWeights(qW, kW)
}
