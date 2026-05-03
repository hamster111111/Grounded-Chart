import unittest
import tempfile
from pathlib import Path

from grounded_chart.api import MatplotlibTraceRunner


class MatplotlibTraceRunnerTest(unittest.TestCase):
    def test_traces_pyplot_bar(self):
        code = """
import matplotlib.pyplot as plt
plt.bar(['A', 'B'], [3, 5])
"""
        trace = MatplotlibTraceRunner().run_code(code)
        self.assertEqual(trace.chart_type, "bar")
        self.assertEqual([(p.x, p.y) for p in trace.points], [("A", 3), ("B", 5)])

    def test_traces_axes_scatter(self):
        code = """
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter([1, 2], [10, 20])
"""
        trace = MatplotlibTraceRunner().run_code(code)
        self.assertEqual(trace.chart_type, "scatter")
        self.assertEqual([(p.x, p.y) for p in trace.points], [(1, 10), (2, 20)])

    def test_traces_pie_labels(self):
        code = """
import matplotlib.pyplot as plt
plt.pie([2, 3], labels=['A', 'B'])
"""
        trace = MatplotlibTraceRunner().run_code(code)
        self.assertEqual(trace.chart_type, "pie")
        self.assertEqual([(p.x, p.y) for p in trace.points], [("A", 2), ("B", 3)])

    def test_traces_figure_metadata(self):
        code = """
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(4, 3))
fig.suptitle("Revenue Dashboard")
ax.plot([1, 2], [3, 4], label="Series A", marker="o")
ax.set_title("Monthly Revenue")
ax.set_xlabel("Month")
ax.set_ylabel("Revenue")
ax.text(1, 3, "peak")
ax.legend()
"""
        run_trace = MatplotlibTraceRunner().run_code_with_figure(code)
        figure = run_trace.figure_trace
        self.assertEqual(figure.title, "Revenue Dashboard")
        self.assertEqual(figure.size_inches, (4.0, 3.0))
        self.assertEqual(figure.axes_count, 1)
        self.assertEqual(figure.axes[0].title, "Monthly Revenue")
        self.assertEqual(figure.axes[0].xlabel, "Month")
        self.assertEqual(figure.axes[0].ylabel, "Revenue")
        self.assertEqual(figure.axes[0].legend_labels, ("Series A",))
        self.assertIn("peak", figure.axes[0].texts)
        self.assertIn("line", [artist.artist_type for artist in figure.axes[0].artists])

    def test_runner_tolerates_savefig_show_and_close(self):
        code = """
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot([1, 2], [3, 4])
plt.savefig('ignored.png')
plt.show()
plt.close(fig)
"""
        run_trace = MatplotlibTraceRunner().run_code_with_figure(code)
        self.assertEqual(run_trace.plot_trace.chart_type, "line")
        self.assertEqual(run_trace.figure_trace.axes_count, 1)

    def test_runner_can_read_relative_data_from_execution_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "data.csv").write_text("x,y\nA,3\nB,5\n", encoding="utf-8")
            code = """
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('data.csv')
plt.bar(df['x'], df['y'])
"""
            trace = MatplotlibTraceRunner().run_code_with_figure(code, execution_dir=root)

        self.assertEqual(trace.plot_trace.chart_type, "bar")
        self.assertEqual([(p.x, p.y) for p in trace.plot_trace.points], [("A", 3), ("B", 5)])

    def test_runner_captures_errorbar_artist_type(self):
        code = """
import numpy as np
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
x = np.array([1, 2, 3])
y = np.array([2, 4, 8])
ax.errorbar(x, y, xerr=0.1 * x, yerr=0.2 * y)
"""
        trace = MatplotlibTraceRunner().run_code_with_figure(code)
        artist_types = [artist.artist_type for artist in trace.figure_trace.axes[0].artists]

        self.assertIn("errorbar", artist_types)

    def test_runner_captures_bar_artist_type(self):
        code = """
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.bar(['A', 'B', 'C'], [3, 5, 4], label='Revenue')
"""
        trace = MatplotlibTraceRunner().run_code_with_figure(code)
        artist_types = [artist.artist_type for artist in trace.figure_trace.axes[0].artists]

        self.assertIn("bar", artist_types)

    def test_runner_captures_pie_artist_type(self):
        code = """
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.pie([35, 45, 20], labels=['Apples', 'Oranges', 'Bananas'])
"""
        trace = MatplotlibTraceRunner().run_code_with_figure(code)
        artist_types = [artist.artist_type for artist in trace.figure_trace.axes[0].artists]

        self.assertIn("pie", artist_types)

    def test_runner_captures_scatter_artist_type(self):
        code = """
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter([1, 2, 3], [4, 5, 6], s=[10, 20, 30])
"""
        trace = MatplotlibTraceRunner().run_code_with_figure(code)
        artist_types = [artist.artist_type for artist in trace.figure_trace.axes[0].artists]

        self.assertIn("scatter", artist_types)

    def test_runner_excludes_colorbar_axes_from_semantic_axes_count(self):
        code = """
import numpy as np
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
image = ax.imshow(np.array([[1, 2], [3, 4]]))
fig.colorbar(image, ax=ax, label='Intensity')
"""
        trace = MatplotlibTraceRunner().run_code_with_figure(code)

        self.assertEqual(1, trace.figure_trace.axes_count)
        self.assertEqual(2, trace.figure_trace.raw["total_axes_count"])
        self.assertEqual(1, trace.figure_trace.raw["helper_axes_count"])
        self.assertEqual(("image",), tuple(artist.artist_type for artist in trace.figure_trace.axes[0].artists))

    def test_runner_exposes_file_path_to_script(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fake_script = root / "script.py"
            code = """
import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.set_title(os.path.basename(__file__))
"""
            trace = MatplotlibTraceRunner().run_code_with_figure(code, execution_dir=root, file_path=fake_script)

        self.assertEqual(trace.figure_trace.axes[0].title, "script.py")

    def test_runner_captures_plotly_figure_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "data.csv").write_text(
                "Major Area,Regions,Country,Overall score\nAsia,East,China,70\nEurope,West,France,80\n",
                encoding="utf-8",
            )
            code = """
import pandas as pd
import plotly.express as px
import plotly.io as pio
df = pd.read_csv('data.csv')
fig = px.sunburst(df, path=['Major Area', 'Regions', 'Country'], values='Overall score', title='Plotly Title')
pio.write_image(fig, 'novice_final.png', width=1000, height=1000)
"""
            trace = MatplotlibTraceRunner().run_code_with_figure(code, execution_dir=root, file_path=root / "plotly_case.py")

        self.assertEqual(trace.plot_trace.source, "plotly_figure")
        self.assertEqual(trace.plot_trace.chart_type, "pie")
        self.assertEqual(trace.figure_trace.source, "plotly_figure")
        self.assertEqual(trace.figure_trace.axes_count, 1)
        self.assertEqual(trace.figure_trace.title, "Plotly Title")
        self.assertEqual(trace.figure_trace.axes[0].projection, "plotly")
        self.assertEqual(trace.figure_trace.axes[0].artists[0].artist_type, "sunburst")

    def test_runner_captures_plotly_annotation_texts(self):
        code = """
import pandas as pd
import plotly.express as px
import plotly.io as pio
df = pd.DataFrame([
    {'Major Area': 'Asia', 'Regions': 'East', 'Country': 'China', 'Overall score': 70},
    {'Major Area': 'Europe', 'Regions': 'West', 'Country': 'France', 'Overall score': 80}
])
fig = px.sunburst(df, path=['Major Area', 'Regions', 'Country'], values='Overall score', title='Plotly Title')
fig.add_annotation(text='Overall score 0-100, 100 = best environment', x=0.5, y=1.02, xref='paper', yref='paper', showarrow=False)
pio.write_image(fig, 'novice_final.png', width=1000, height=1000)
"""
        trace = MatplotlibTraceRunner().run_code_with_figure(code)

        self.assertIn("Overall score 0-100, 100 = best environment", trace.figure_trace.axes[0].texts)

    def test_runner_captures_plotly_axis_and_legend_metadata(self):
        code = """
import pandas as pd
import plotly.express as px
import plotly.io as pio
df = pd.DataFrame([
    {'category': 'A', 'sales': 10, 'segment': 'North'},
    {'category': 'B', 'sales': 7, 'segment': 'South'}
])
fig = px.bar(df, x='category', y='sales', color='segment', title='Plotly Bar')
fig.update_layout(xaxis_title='Category', yaxis_title='Sales')
pio.write_image(fig, 'novice_final.png', width=1000, height=1000)
"""
        trace = MatplotlibTraceRunner().run_code_with_figure(code)
        axis = trace.figure_trace.axes[0]

        self.assertEqual("Category", axis.xlabel)
        self.assertEqual("Sales", axis.ylabel)
        self.assertIn("North", axis.legend_labels)
        self.assertIn("South", axis.legend_labels)

    def test_runner_captures_plotly_trace_texts(self):
        code = """
import plotly.graph_objects as go
fig = go.Figure()
fig.add_bar(x=['A', 'B'], y=[10, 7], text=['peak', 'low'])
fig.write_image('novice_final.png', width=1000, height=1000)
"""
        trace = MatplotlibTraceRunner().run_code_with_figure(code)

        self.assertIn("peak", trace.figure_trace.axes[0].texts)
        self.assertIn("low", trace.figure_trace.axes[0].texts)

    def test_runner_captures_plotly_facet_axes_count(self):
        code = """
import pandas as pd
import plotly.express as px
import plotly.io as pio
df = pd.DataFrame([
    {'category': 'A', 'sales': 10, 'region': 'East'},
    {'category': 'B', 'sales': 7, 'region': 'East'},
    {'category': 'A', 'sales': 6, 'region': 'West'},
    {'category': 'B', 'sales': 9, 'region': 'West'}
])
fig = px.bar(df, x='category', y='sales', facet_col='region', title='Facet Sales')
pio.write_image(fig, 'plotly_facet_axes.png', width=1000, height=1000)
"""
        trace = MatplotlibTraceRunner().run_code_with_figure(code)

        self.assertEqual(2, trace.figure_trace.axes_count)

    def test_runner_captures_plotly_shape_artist_count(self):
        code = """
import plotly.graph_objects as go
fig = go.Figure()
fig.add_scatter(x=[1, 2], y=[3, 4], mode='lines', name='Series')
fig.add_shape(type='line', x0=1, x1=2, y0=3.5, y1=3.5)
fig.write_image('plotly_shape_artist.png', width=1000, height=1000)
"""
        trace = MatplotlibTraceRunner().run_code_with_figure(code)
        artist_counts = {artist.artist_type: artist.count for artist in trace.figure_trace.axes[0].artists}

        self.assertEqual(1, artist_counts.get("shape"))

    def test_runner_captures_plotly_colorbar_title_text(self):
        code = """
import plotly.graph_objects as go
fig = go.Figure(data=go.Heatmap(z=[[1, 2], [3, 4]], colorbar=dict(title='Intensity')))
fig.write_image('plotly_colorbar_title.png', width=1000, height=1000)
"""
        trace = MatplotlibTraceRunner().run_code_with_figure(code)

        self.assertIn("Intensity", trace.figure_trace.axes[0].texts)

    def test_runner_extracts_plotly_bar_points(self):
        code = """
import plotly.graph_objects as go
fig = go.Figure()
fig.add_bar(x=['A', 'B'], y=[10, 7])
fig.write_image('plotly_bar_points.png', width=1000, height=1000)
"""
        trace = MatplotlibTraceRunner().run_code_with_figure(code)

        self.assertEqual("bar", trace.plot_trace.chart_type)
        self.assertEqual([("A", 10), ("B", 7)], [(point.x, point.y) for point in trace.plot_trace.points])

    def test_runner_extracts_plotly_line_points_from_scatter_mode(self):
        code = """
import plotly.graph_objects as go
fig = go.Figure()
fig.add_scatter(x=['Jan', 'Feb'], y=[3, 5], mode='lines')
fig.write_image('plotly_line_points.png', width=1000, height=1000)
"""
        trace = MatplotlibTraceRunner().run_code_with_figure(code)

        self.assertEqual("line", trace.plot_trace.chart_type)
        self.assertEqual([("Jan", 3), ("Feb", 5)], [(point.x, point.y) for point in trace.plot_trace.points])

    def test_runner_extracts_plotly_pie_points(self):
        code = """
import plotly.graph_objects as go
fig = go.Figure(data=go.Pie(labels=['A', 'B'], values=[2, 3]))
fig.write_image('plotly_pie_points.png', width=1000, height=1000)
"""
        trace = MatplotlibTraceRunner().run_code_with_figure(code)

        self.assertEqual("pie", trace.plot_trace.chart_type)
        self.assertEqual([("A", 2), ("B", 3)], [(point.x, point.y) for point in trace.plot_trace.points])

    def test_runner_extracts_plotly_grouped_bar_points_with_series_keys(self):
        code = """
import plotly.graph_objects as go
fig = go.Figure()
fig.add_bar(name='North', x=['A', 'B'], y=[10, 7])
fig.add_bar(name='South', x=['A', 'B'], y=[6, 9])
fig.write_image('plotly_grouped_bar_points.png', width=1000, height=1000)
"""
        trace = MatplotlibTraceRunner().run_code_with_figure(code)

        self.assertEqual("bar", trace.plot_trace.chart_type)
        self.assertEqual(
            [(("A", "North"), 10), (("B", "North"), 7), (("A", "South"), 6), (("B", "South"), 9)],
            [(point.x, point.y) for point in trace.plot_trace.points],
        )

    def test_runner_extracts_plotly_multi_line_points_with_series_keys(self):
        code = """
import plotly.graph_objects as go
fig = go.Figure()
fig.add_scatter(name='North', x=['Jan', 'Feb'], y=[3, 5], mode='lines')
fig.add_scatter(name='South', x=['Jan', 'Feb'], y=[4, 6], mode='lines')
fig.write_image('plotly_multi_line_points.png', width=1000, height=1000)
"""
        trace = MatplotlibTraceRunner().run_code_with_figure(code)

        self.assertEqual("line", trace.plot_trace.chart_type)
        self.assertEqual(
            [(("Jan", "North"), 3), (("Feb", "North"), 5), (("Jan", "South"), 4), (("Feb", "South"), 6)],
            [(point.x, point.y) for point in trace.plot_trace.points],
        )

    def test_runner_extracts_plotly_multi_scatter_points_with_series_keys(self):
        code = """
import plotly.graph_objects as go
fig = go.Figure()
fig.add_scatter(name='North', x=[1, 2], y=[3, 5], mode='markers')
fig.add_scatter(name='South', x=[1, 2], y=[4, 6], mode='markers')
fig.write_image('plotly_multi_scatter_points.png', width=1000, height=1000)
"""
        trace = MatplotlibTraceRunner().run_code_with_figure(code)

        self.assertEqual("scatter", trace.plot_trace.chart_type)
        self.assertEqual(
            [((1, "North"), 3), ((2, "North"), 5), ((1, "South"), 4), ((2, "South"), 6)],
            [(point.x, point.y) for point in trace.plot_trace.points],
        )

    def test_runner_extracts_plotly_grouped_facet_bar_points_with_facet_keys(self):
        code = """
import pandas as pd
import plotly.express as px
import plotly.io as pio
df = pd.DataFrame([
    {'category': 'A', 'segment': 'Retail', 'region': 'East', 'sales': 10},
    {'category': 'B', 'segment': 'Retail', 'region': 'East', 'sales': 7},
    {'category': 'A', 'segment': 'Wholesale', 'region': 'East', 'sales': 6},
    {'category': 'B', 'segment': 'Wholesale', 'region': 'East', 'sales': 9},
    {'category': 'A', 'segment': 'Retail', 'region': 'West', 'sales': 11},
    {'category': 'B', 'segment': 'Retail', 'region': 'West', 'sales': 8},
    {'category': 'A', 'segment': 'Wholesale', 'region': 'West', 'sales': 5},
    {'category': 'B', 'segment': 'Wholesale', 'region': 'West', 'sales': 4}
])
fig = px.bar(df, x='category', y='sales', color='segment', facet_col='region', barmode='group')
pio.write_image(fig, 'plotly_grouped_facet_bar_points.png', width=1000, height=1000)
"""
        trace = MatplotlibTraceRunner().run_code_with_figure(code)

        self.assertEqual("bar", trace.plot_trace.chart_type)
        self.assertIn((("A", "Retail", "East"), 10), [(point.x, point.y) for point in trace.plot_trace.points])
        self.assertIn((("B", "Wholesale", "West"), 4), [(point.x, point.y) for point in trace.plot_trace.points])
        self.assertIn("region=East", trace.figure_trace.axes[0].texts)

    def test_runner_extracts_plotly_stacked_facet_bar_points_with_facet_keys(self):
        code = """
import pandas as pd
import plotly.express as px
import plotly.io as pio
df = pd.DataFrame([
    {'category': 'A', 'segment': 'Retail', 'region': 'East', 'sales': 10},
    {'category': 'B', 'segment': 'Retail', 'region': 'East', 'sales': 7},
    {'category': 'A', 'segment': 'Wholesale', 'region': 'East', 'sales': 6},
    {'category': 'B', 'segment': 'Wholesale', 'region': 'East', 'sales': 9},
    {'category': 'A', 'segment': 'Retail', 'region': 'West', 'sales': 11},
    {'category': 'B', 'segment': 'Retail', 'region': 'West', 'sales': 8},
    {'category': 'A', 'segment': 'Wholesale', 'region': 'West', 'sales': 5},
    {'category': 'B', 'segment': 'Wholesale', 'region': 'West', 'sales': 4}
])
fig = px.bar(df, x='category', y='sales', color='segment', facet_col='region', barmode='stack')
pio.write_image(fig, 'plotly_stacked_facet_bar_points.png', width=1000, height=1000)
"""
        trace = MatplotlibTraceRunner().run_code_with_figure(code)

        self.assertEqual("bar", trace.plot_trace.chart_type)
        self.assertIn((("A", "Retail", "East"), 10), [(point.x, point.y) for point in trace.plot_trace.points])
        self.assertIn((("B", "Wholesale", "West"), 4), [(point.x, point.y) for point in trace.plot_trace.points])
        self.assertIn("region=East", trace.figure_trace.axes[0].texts)


if __name__ == "__main__":
    unittest.main()
