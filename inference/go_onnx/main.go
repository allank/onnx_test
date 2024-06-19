package main

import (
	"encoding/csv"
	"fmt"
	ort "github.com/yalue/onnxruntime_go"
	"os"
	"strconv"
	"time"
)

// Calculates the accuracy between y (ground truth) and y_pred (predicted values)
// Simple proportion of matching results
func accuracy(y, y_pred []int64) float64 {
	total := float64(len(y))
	match := float64(0)
	for i := range y {
		if y[i] == y_pred[i] {
			match = match + 1
		}
	}
	if match == 0 {
		return float64(0)
	}
	return float64(match / total)
}

func main() {

	lTime := time.Now()

	// Read training data and ground truth
	csvFile, err := os.Open("../../data/X_test.csv")
	if err != nil {
		fmt.Println("Could not read test data")
		return
	}
	defer csvFile.Close()
	csvReader := csv.NewReader(csvFile)
	X, _ := csvReader.ReadAll()
	csvFile.Close()

	csvFile, err = os.Open("../../data/y_test.csv")
	if err != nil {
		fmt.Println("Could not read ground truth data")
		return
	}
	defer csvFile.Close()
	csvReader = csv.NewReader(csvFile)
	y, _ := csvReader.ReadAll()
	csvFile.Close()

	// This needs to be set to the appropriate runtime for your platform
	ort.SetSharedLibraryPath("libonnxruntime.1.18.0.dylib")

	err = ort.InitializeEnvironment()
	defer ort.DestroyEnvironment()
	if err != nil {
		fmt.Println("Error starting environment: ", err)
	}

	// input tensor needs to be rank 1
	// capture number of rows for later
	rows := int64(len(X))
	inputData := []float32{}

	// Flatten matrix, storing inputs as float32
	// TODO: we should probably exit if we cannot parse any of the values
	for _, row := range X {
		for _, col := range row {
			if f, err := strconv.ParseFloat(col, 32); err == nil {
				inputData = append(inputData, float32(f))
			}
		}
	}

	// store ground truth as int64
	// TODO: we should probably exit if we cannot parse any of the values
	y_truth := []int64{}
	for _, i := range y {
		if i64, err := strconv.ParseFloat(i[0], 32); err == nil {
			y_truth = append(y_truth, int64(i64))
		}
	}

	// ONNX runtime needs to know the original shape of our data
	// This is the number of rows x 4 features fitted in original model
	inputShape := ort.NewShape(rows, 4)
	inputTensor, err := ort.NewTensor(inputShape, inputData)
	if err != nil {
		fmt.Println("Could not create input tensor:", err)
	}
	defer inputTensor.Destroy()

	// For this model it is rank 1
	outputShape := ort.NewShape(rows)
	outputTensor, err := ort.NewEmptyTensor[int64](outputShape)
	if err != nil {
		fmt.Println("Could not create output tensor:", err)
	}
	defer outputTensor.Destroy()

	sTime := time.Now()

	// TODO: Separate the loading of the model from the inference
	// Timings here may be skewed by needing to instantiate the model as part of the inference process
	session, err := ort.NewAdvancedSession("../../data/rf_iris.onnx",
		[]string{"X"}, []string{"output_label"},
		[]ort.ArbitraryTensor{inputTensor}, []ort.ArbitraryTensor{outputTensor}, nil)
	defer session.Destroy()
	if err != nil {
		fmt.Println("Error creating session: ", err)
	}

	err = session.Run()
	if err != nil {
		fmt.Println("Error running session: ", err)
	}

	outputData := outputTensor.GetData()

	eTime := time.Now()
	fmt.Println("Accuracy: ", accuracy(y_truth, outputData))
	fmt.Println("Total run time: ", eTime.Sub(lTime))
	fmt.Println("Inference time: ", eTime.Sub(sTime))
}
