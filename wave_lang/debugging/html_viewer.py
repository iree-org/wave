import os
from typing import Dict, Any
import json


def html_viewer(debug_logs: Dict[str, Any]) -> None:
    """
    Create a debug-view.html file with the provided dictionary data embedded as JavaScript.

    Args:
        data: The debug_logs dictionary to embed in the HTML file for viewing.
        Its shape is given in wave_ops.py in the docstring for `debug_log`.
    """

    def jsonify_debug_logs(debug_logs: Dict[str, Any]):
        def jsonify_debug_log(log: Dict[str, Any]):
            serializable = {k: v for k, v in log.items()}
            # convert torch tensor to serializable list
            serializable["value"] = serializable["value"].tolist()
            serializable["symbolic_shape"] = [
                str(symbol) for symbol in serializable["symbolic_shape"]
            ]
            return serializable

        jsonable = {label: jsonify_debug_log(log) for label, log in debug_logs.items()}
        return json.dumps(jsonable)

    def get_sibling_file(filename):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    html_string = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Debug Data Viewer</title>
    <script crossorigin src="https://unpkg.com/react@18/umd/react.development.js"></script>
    <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <style>
        {get_sibling_file('html_viewer.css')}
    </style>
</head>
<body>
    <div id="root"></div>

    <script>
        const debug_logs = {jsonify_debug_logs(debug_logs)};
    </script>

    <script>
        {get_sibling_file('html_viewer.js')}
    </script>

</body>
</html>
"""
    output_path = "debug-view.html"
    try:
        with open(output_path, "w", encoding="utf-8") as html_file:
            html_file.write(html_string)
        print(f"Debug view saved to: {output_path}")
    except Exception as e:
        print(f"Error writing HTML file: {str(e)}")
