// React-based tensor viewer implementation

const {useState, useEffect, useCallback} = React;

// Parse dimension view string like "M, N" or "A=0, B=1, C, D"
function parseDimensionView(viewStr, symbolicShape) {
  const parts = viewStr.split(",").map((s) => s.trim()).filter((s) => s);
  const result = {
    fixed : {},
    viewDims : [],
  };

  for (const part of parts) {
    if (part.includes("=")) {
      const [dim, value] = part.split("=").map((s) => s.trim());

      if (!dim) {
        throw new Error(`Missing dimension name before '=' in "${part}"`);
      }
      if (value === "" || value === undefined) {
        throw new Error(
            `Missing value after '=' in "${part}". Expected format: ${
                dim}=<number>`,
        );
      }
      const numValue = parseInt(value);
      if (isNaN(numValue)) {
        throw new Error(
            `Invalid number "${value}" in "${part}". Expected format: ${
                dim}=<number>`,
        );
      }

      result.fixed[dim] = numValue;
    } else {
      if (!part) {
        throw new Error("Empty dimension name found");
      }
      result.viewDims.push(part);
    }
  }

  return result;
}

function getDefaultViewString(symbolicShape) {
  if (!symbolicShape || symbolicShape.length === 0) {
    return "";
  }

  if (symbolicShape.length <= 2) {
    return symbolicShape.join(", ");
  } else {
    // For >2 dims, fix outer dimensions to 0
    const fixed = symbolicShape.slice(0, -2).map((dim) => `${dim}=0`);
    const viewDims = symbolicShape.slice(-2);
    return [...fixed, ...viewDims ].join(", ");
  }
}

// Get tensor dimensions based on nested array structure
function getTensorShape(tensor) {
  if (!Array.isArray(tensor))
    return [];

  const shape = [ tensor.length ];
  let current = tensor[0];

  while (Array.isArray(current)) {
    shape.push(current.length);
    current = current[0];
  }

  return shape;
}

// Apply dimension fixing and view transformation to tensor data
function applyDimensionView(tensorData, symbolicShape, viewStr) {
  if (!tensorData || !symbolicShape) {
    return {data : null, error : "Bad debug log data."};
  }

  try {
    const parsedDimensionView = parseDimensionView(viewStr, symbolicShape);

    if (parsedDimensionView.viewDims.length > 2) {
      return {
        data : null,
        error : "Cannot view more than 2 dimensions at once",
      };
    }

    const dimensionNameToIndex = {};
    symbolicShape.forEach((dim, idx) => { dimensionNameToIndex[dim] = idx; });

    for (const dim of parsedDimensionView.viewDims) {
      if (!(dim in dimensionNameToIndex)) {
        return {
          data : null,
          error : `Unknown dimension '${dim}'. Available: ${
              symbolicShape.join(", ")}`,
        };
      }
    }

    for (const dim of Object.keys(parsedDimensionView.fixed)) {
      if (!(dim in dimensionNameToIndex)) {
        return {
          data : null,
          error : `Unknown dimension '${dim}' in fixed value. Available: ${
              symbolicShape.join(", ")}`,
        };
      }
    }

    // Transform the tensor data
    const transformedData = transformTensorData(
        tensorData,
        symbolicShape,
        parsedDimensionView,
        dimensionNameToIndex,
    );

    return {data : transformedData, error : null};
  } catch (error) {
    return {
      data : null,
      error : `Invalid view string: ${error.message}`,
    };
  }
}

// Transform tensor data based on fixed dimensions and view dimensions
function transformTensorData(
    tensor,
    symbolicShape,
    parsedDimensionView,
    dimensionNameToIndex,
) {
  // Apply fixed dimensions first - slice the tensor at fixed indices for fixed
  // dimensions
  let slicedTensor = tensor;

  // Sort fixed dimensions by their original dimension index to apply them in
  // the right order
  const sortedFixed = Object.entries(parsedDimensionView.fixed)
                          .map(([ dimName, fixedValue ]) => ({
                                 dimName,
                                 fixedValue,
                                 dimIndex : dimensionNameToIndex[dimName],
                               }))
                          .sort();

  // Apply fixed dimensions from outermost to innermost dimension
  for (const {dimName, fixedValue, dimIndex} of sortedFixed) {
    // Calculate the adjusted dimension index after previous fixed dimensions
    const adjustedDimIndex =
        dimIndex - sortedFixed.filter((c) => c.dimIndex < dimIndex).length;
    slicedTensor = sliceAtDimension(slicedTensor, adjustedDimIndex, fixedValue);
  }

  if (parsedDimensionView.viewDims.length === 0 ||
      parsedDimensionView.viewDims.length === 1) {
    return slicedTensor;
  } else if (parsedDimensionView.viewDims.length === 2) {
    // Two dimension view - need to check for transposition
    return finalize2DView(
        slicedTensor,
        parsedDimensionView,
        dimensionNameToIndex,
        symbolicShape,
    );
  }

  return slicedTensor;
}

// Slice tensor at a specific dimension and index
function sliceAtDimension(tensor, dimIndex, sliceIndex) {
  if (dimIndex === 0) {
    if (Array.isArray(tensor) && tensor.length > sliceIndex) {
      return tensor[sliceIndex];
    } else {
      throw new Error("Bad fixed value for dimension");
    }
  }

  if (!Array.isArray(tensor)) {
    throw new Error(
        "Error slicing for fixed dimensions, but this shouldn't happen...",
    );
  }

  return tensor.map(
      (subArray) => sliceAtDimension(subArray, dimIndex - 1, sliceIndex),
  );
}

// Ensure data is formatted as 2D view with proper row/column mapping
function finalize2DView(
    tensor,
    parsedDimensionView,
    dimensionNameToIndex,
    symbolicShape,
) {
  const shape = getTensorShape(tensor);
  if (shape.length != 2) {
    throw new Error("finalize2DView should only be getting 2d data");
  }

  const rowDimName = parsedDimensionView.viewDims[0];
  const colDimName = parsedDimensionView.viewDims[1];

  // Find the original dimension order to determine whether to transpose.
  const originalRowIndex = dimensionNameToIndex[rowDimName];
  const originalColIndex = dimensionNameToIndex[colDimName];
  const fixedDimIndices = Object.keys(parsedDimensionView.fixed)
                              .map((dimName) => dimensionNameToIndex[dimName])
                              .sort();
  const remainingDimIndices =
      symbolicShape.map((_, idx) => idx)
          .filter((idx) => !fixedDimIndices.includes(idx));
  const newRowIndex = remainingDimIndices.indexOf(
      dimensionNameToIndex[rowDimName],
  );
  const newColIndex = remainingDimIndices.indexOf(
      dimensionNameToIndex[colDimName],
  );
  if (newRowIndex === -1 || newColIndex === -1) {
    throw new Error("error showing un-fixed dimensions");
  }
  if (newRowIndex < newColIndex) {
    return tensor;
  } else {
    return transpose2D(tensor);
  }
}

function transpose2D(matrix) {
  if (!Array.isArray(matrix) || matrix.length === 0)
    return matrix;
  if (!Array.isArray(matrix[0]))
    return matrix;

  const rows = matrix.length;
  const cols = matrix[0].length;
  const result = [];

  for (let j = 0; j < cols; j++) {
    const newRow = [];
    for (let i = 0; i < rows; i++) {
      newRow.push(matrix[i][j]);
    }
    result.push(newRow);
  }

  return result;
}

function formatTensorDisplay(tensor, viewMode) {
  if (!tensor) {
    return String(tensor);
  }

  if (viewMode === "compressed-global") {
    return formatCompressedView(tensor);
  } else {
    return formatFullView(tensor);
  }
}

function formatCompressedView(tensor) {
  // Scalar
  if (!Array.isArray(tensor))
    return formatNumber(tensor);

  // 1D
  if (!Array.isArray(tensor[0])) {
    if (tensor.length <= 6) {
      return "[" + tensor.map(formatNumber).join(", ") + "]";
    }
    const first3 = tensor.slice(0, 3).map(formatNumber);
    const last3 = tensor.slice(-3).map(formatNumber);
    return "[" + first3.join(", ") + ", ..., " + last3.join(", ") + "]";
  }

  // 2D
  const formatRow = (row) => {
    if (row.length <= 6) {
      return "[" + row.map(formatNumber).join(", ") + "]";
    }
    const first3 = row.slice(0, 3).map(formatNumber);
    const last3 = row.slice(-3).map(formatNumber);
    return "[" + first3.join(", ") + ", ..., " + last3.join(", ") + "]";
  };

  if (tensor.length <= 6) {
    return "[\n  " + tensor.map(formatRow).join(",\n  ") + "\n]";
  }

  const first3 = tensor.slice(0, 3).map(formatRow);
  const last3 = tensor.slice(-3).map(formatRow);
  return ("[\n  " + first3.join(",\n  ") + ",\n  ...,\n  " +
          last3.join(",\n  ") + "\n]");
}

function formatFullView(tensor) {
  if (!Array.isArray(tensor))
    return formatNumber(tensor);

  if (!Array.isArray(tensor[0])) {
    return "[" + tensor.map(formatNumber).join(", ") + "]";
  }

  const formatRow = (row) => "[" + row.map(formatNumber).join(", ") + "]";
  return "[\n  " + tensor.map(formatRow).join(",\n  ") + "\n]";
}

function formatNumber(num) {
  if (typeof num !== "number")
    return String(num);
  if (Number.isInteger(num))
    return num.toString();
  return num.toFixed(4);
}

// Each debug_log item gets a TensorViewer
function TensorViewer({label, tensorInfo}) {
  const [viewString, setViewString] = useState("");
  const [viewMode, setViewMode] = useState("compressed-global");
  const [displayData, setDisplayData] = useState(null);
  const [error, setError] = useState(null);

  // Initialize default view string and perform initial display update
  useEffect(() => {
    if (tensorInfo.symbolic_shape) {
      const defaultView = getDefaultViewString(tensorInfo.symbolic_shape);
      setViewString(defaultView);

      if (!tensorInfo.value) {
        setError("No tensor data available");
        return;
      }

      const result = applyDimensionView(
          tensorInfo.value,
          tensorInfo.symbolic_shape,
          defaultView,
      );

      if (result.error) {
        setError(result.error);
        setDisplayData(null);
      } else {
        setError(null);
        setDisplayData(result.data);
      }
    }
  }, [ tensorInfo.symbolic_shape, tensorInfo.value ]);

  const updateDisplay = useCallback(() => {
    if (!tensorInfo.value) {
      setError("No tensor data available");
      return;
    }

    const result = applyDimensionView(
        tensorInfo.value,
        tensorInfo.symbolic_shape,
        viewString,
    );

    if (result.error) {
      setError(result.error);
      setDisplayData(null);
    } else {
      setError(null);
      setDisplayData(result.data);
    }
  }, [ tensorInfo, viewString ]);

  return React.createElement(
      "div",
      {className : "tensor-viewer"},
      React.createElement("div", {className : "tensor-label"}, label),

      React.createElement(
          "div",
          {className : "tensor-shape-info"},
          `Shape: [${tensorInfo.symbolic_shape.join(", ")}]`,
          ),

      React.createElement(
          "div",
          {className : "controls"},
          React.createElement("input", {
            type : "text",
            className : "view-input",
            value : viewString,
            onChange : (e) => { setViewString(e.target.value); },
            onKeyDown : (e) => {
              if (e.key === "Enter") {
                updateDisplay();
              }
            },
            placeholder : "e.g., M, N or A=0, B=1, C, D",
          }),
          React.createElement(
              "button",
              {
                className : "update-button",
                onClick : updateDisplay,
              },
              "Update",
              ),
          React.createElement(
              "select",
              {
                className : "view-mode-select",
                value : viewMode,
                onChange : (e) => { setViewMode(e.target.value); },
              },
              React.createElement(
                  "option",
                  {value : "compressed-global"},
                  "Compressed Global",
                  ),
              React.createElement("option", {value : "full-global"},
                                  "Full Global"),
              ),
          ),

      error ? React.createElement("div", {className : "error"}, error)
            : React.createElement(
                  "div",
                  {className : "tensor-display"},
                  formatTensorDisplay(displayData, viewMode),
                  ),
  );
}

// Main app component
function DebugDataViewer({debug_logs}) {
  if (!debug_logs || typeof debug_logs !== "object") {
    return React.createElement(
        "div",
        {className : "error"},
        "No debug data available or invalid data format",
    );
  }

  const tensorEntries = Object.entries(debug_logs);
  return React.createElement(
      "div",
      null,
      React.createElement("h1", null, "Debug Data Viewer"),
      ...tensorEntries.map(
          ([ label, tensorInfo ]) => React.createElement(TensorViewer, {
            key : label,
            label : label,
            tensorInfo : tensorInfo,
          }),
          ),
  );
}

function initializeApp() {
  const root = ReactDOM.createRoot(document.getElementById("root"));
  try {
    root.render(React.createElement(DebugDataViewer, {debug_logs}));
  } catch (error) {
    console.error("Error rendering app:", error);
    root.render(
        React.createElement(
            "div",
            {className : "error"},
            `Error rendering viewer: ${error.message}`,
            ),
    );
  }
}

// Start the app when the page loads
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initializeApp);
} else {
  initializeApp();
}
