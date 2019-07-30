from IPython.core.display import display, HTML
import json
import numpy as np


def plot3D(X, Y, Z, height=600, xlabel="X", ylabel="Y", zlabel="Z", initialCamera=None):
    """
    Code from
    https://stackoverflow.com/questions/38364435/python-matplotlib-make-3d-plot-interactive-in-jupyter-notebook
    See also
    https://visjs.github.io/vis-graph3d/examples/graph3d/playground/index.html
    """
    options = {
        "width": "100%",
        "style": "surface",
        "showPerspective": True,
        "showGrid": True,
        "showShadow": False,
        "keepAspectRatio": True,
        "height": str(height) + "px"
    }

    if initialCamera:
        options["cameraPosition"] = initialCamera

    data = [{"x": X[y, x], "y": Y[y, x], "z": Z[y, x]} for y in range(X.shape[0]) for x in range(X.shape[1])]
    visCode = r"""
       <link href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css" type="text/css" rel="stylesheet" />
       <script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
       <div id="pos" style="top:0px;left:0px;position:absolute;"></div>
       <div id="visualization"></div>
       <script type="text/javascript">
        var data = new vis.DataSet();
        data.add(""" + json.dumps(data) + """);
        var options = """ + json.dumps(options) + """;
        var container = document.getElementById("visualization");
        var graph3d = new vis.Graph3d(container, data, options);
        graph3d.on("cameraPositionChange", function(evt)
        {
            elem = document.getElementById("pos");
            elem.innerHTML = "H: " + evt.horizontal + "<br>V: " + evt.vertical + "<br>D: " + evt.distance;
        });
       </script>
    """
    htmlCode = "<iframe srcdoc='" + visCode + "' width='100%' height='" + str(
        height) + "px' style='border:0;' scrolling='no'> </iframe>"
    display(HTML(htmlCode))


def run_plot3d_nb():
    X, Y = np.meshgrid(np.linspace(-3,3,50),np.linspace(-3,3,50))
    Z = np.sin(X**2 + Y**2)**2/(X**2+Y**2)
    plot3D(X, Y, Z)
