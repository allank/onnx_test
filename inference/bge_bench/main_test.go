package main

import "testing"

func TestMaxAbsDiff(t *testing.T) {
	cases := []struct {
		name string
		a, b []float64
		want float64
	}{
		{
			name: "identical vectors",
			a:    []float64{0.1, -0.2, 0.3},
			b:    []float64{0.1, -0.2, 0.3},
			want: 0,
		},
		{
			name: "single differing element",
			a:    []float64{1.0, 2.0, 3.0},
			b:    []float64{1.0, 2.5, 3.0},
			want: 0.5,
		},
		{
			name: "negative difference reported as absolute value",
			a:    []float64{1.0, 2.0},
			b:    []float64{1.0, 2.7},
			want: 0.7,
		},
		{
			name: "largest of several differences wins",
			a:    []float64{0.0, 0.0, 0.0},
			b:    []float64{0.01, -0.2, 0.05},
			want: 0.2,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := maxAbsDiff(tc.a, tc.b)
			const eps = 1e-12
			if diff := got - tc.want; diff < -eps || diff > eps {
				t.Errorf("maxAbsDiff(%v, %v) = %v, want %v", tc.a, tc.b, got, tc.want)
			}
		})
	}
}
