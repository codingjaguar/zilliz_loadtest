package validation

import (
	"testing"
)

func TestPrintValidationResults(t *testing.T) {
	// This is a basic test to ensure the function doesn't panic
	result := &ValidationResult{
		CollectionExists:   true,
		CollectionLoaded:   true,
		HasData:            true,
		RowCount:           1000000,
		VectorFieldExists:  true,
		VectorDimension:    768,
		IndexExists:        true,
		IndexBuilt:         true,
		ConnectionHealthy:  true,
		Errors:             []string{},
		Warnings:           []string{},
	}

	// Just verify it doesn't panic
	PrintValidationResults(result, "test_collection")
}

func TestPrintValidationResultsWithErrors(t *testing.T) {
	result := &ValidationResult{
		CollectionExists:   false,
		CollectionLoaded:   false,
		HasData:            false,
		VectorFieldExists:  false,
		IndexExists:        false,
		ConnectionHealthy:  false,
		Errors:             []string{"Collection does not exist", "No vector field found"},
		Warnings:           []string{"Collection may not be loaded"},
	}

	// Just verify it doesn't panic
	PrintValidationResults(result, "test_collection")
}
