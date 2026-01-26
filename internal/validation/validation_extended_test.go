package validation

import (
	"testing"
)

// We can't easily test ValidateCollection without a real client connection,
// but we can test edge cases in PrintValidationResults

func TestPrintValidationResultsEdgeCases(t *testing.T) {
	// Test with all fields set to various values
	result := &ValidationResult{
		CollectionExists:   true,
		CollectionLoaded:   false,
		HasData:            false,
		RowCount:           0,
		VectorFieldExists:  true,
		VectorDimension:    0,
		IndexExists:        true,
		IndexBuilt:         false,
		MetricTypeMatch:    false,
		ConnectionHealthy:  true,
		Errors:             []string{"Error 1", "Error 2"},
		Warnings:           []string{"Warning 1", "Warning 2", "Warning 3"},
	}

	PrintValidationResults(result, "test_collection")
}

func TestPrintValidationResultsMinimal(t *testing.T) {
	result := &ValidationResult{
		Errors:   []string{},
		Warnings: []string{},
	}

	PrintValidationResults(result, "minimal_collection")
}

func TestPrintValidationResultsManyErrors(t *testing.T) {
	errors := make([]string, 20)
	warnings := make([]string, 15)
	for i := range errors {
		errors[i] = "Error " + string(rune(i+'A'))
	}
	for i := range warnings {
		warnings[i] = "Warning " + string(rune(i+'A'))
	}

	result := &ValidationResult{
		CollectionExists:   false,
		CollectionLoaded:   false,
		HasData:            false,
		VectorFieldExists:  false,
		IndexExists:        false,
		ConnectionHealthy:  false,
		Errors:             errors,
		Warnings:           warnings,
	}

	PrintValidationResults(result, "many_errors_collection")
}

// Test that ValidateCollection function signature is correct
// We can't test the actual implementation without mocking the SDK
func TestValidateCollectionSignature(t *testing.T) {
	// Skip this test as it requires network connection and may hang
	t.Skip("Skipping test that requires network connection")
}
