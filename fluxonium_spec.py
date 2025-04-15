import numpy as np
import scqubits as scq
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.express.colors import qualitative

# 模擬參數
evals_count = 6
flux_vals = np.linspace(0, 1, 101)

# 模擬 Fluxonium
q1 = scq.Fluxonium(EJ=3.7, EC=0.94, EL=0.59, flux=0.5, cutoff=51)
spec = q1.get_spectrum_vs_paramvals('flux', flux_vals, evals_count).energy_table
mtrix = q1.get_matelements_vs_paramvals('n_operator', 'flux', flux_vals, evals_count).matrixelem_table

# Transition 組合
transitions = [(n, n+1) for n in range(evals_count - 1)] + [(n, n+2) for n in range(evals_count - 2)]
transition_labels = [f"|{i}⟩→|{j}⟩" for (i, j) in transitions]
label_to_pair = dict(zip(transition_labels, transitions))

# 顏色盤
palette = qualitative.Dark2
color_cycle = palette * (len(transitions) // len(palette) + 1)
label_to_color = dict(zip(transition_labels, color_cycle))

# 畫 segment-trace 函數
def plot_transition_segments(x, y, alpha, color='0,0,0', name=None, matrix_element=True):
    seg_traces = []
    for i in range(len(x) - 1):
        x_seg = [x[i], x[i+1]]
        y_seg = [y[i], y[i+1]]
        a = alpha[i]
        rgba = f'rgba({color},{a:.3f})'
        trace = go.Scatter(
            x=x_seg,
            y=y_seg,
            mode='lines',
            line=dict(color=rgba, width=2),
            showlegend=(i == 0),
            name=name if i == 0 else None,
            hoverinfo='x+y+text',
            text=[f'α={a:.3f}']*2 if matrix_element else None
        )
        seg_traces.append(trace)
    return seg_traces

# Dash App
app = Dash(__name__)
app.layout = html.Div([
    html.H3("Fluxonium Spectrum - Segment Style"),

    html.Label("Select Transitions:"),
    dcc.Dropdown(
        options=[{"label": lbl, "value": lbl} for lbl in transition_labels],
        value=transition_labels[:4],
        multi=True,
        id='transition-select'
    ),

    html.Label("Use Matrix Element for Opacity:"),
    dcc.Checklist(
        options=[{"label": "Enable", "value": "on"}],
        value=["on"],
        id='matrix-toggle'
    ),

    dcc.Graph(id='fluxonium-plot')
])

@app.callback(
    Output('fluxonium-plot', 'figure'),
    Input('transition-select', 'value'),
    Input('matrix-toggle', 'value')
)
def update_plot(selected_transitions, matrix_flag):
    matrix_element = "on" in matrix_flag
    traces = []

    for label in selected_transitions:
        i, j = label_to_pair[label]
        freq_ij = spec[:, j] - spec[:, i]
        color_rgb = label_to_color[label].replace("rgb(", "").replace(")", "")
        
        if matrix_element:
            m_ij = np.abs(mtrix[:, i, j])
            norm_alpha = (m_ij - m_ij.min()) / (m_ij.max() - m_ij.min() + 1e-12)
            alpha = 0.3 + 0.7 * norm_alpha
        else:
            alpha = np.ones_like(freq_ij)

        seg = plot_transition_segments(flux_vals, freq_ij, alpha, color=color_rgb, name=label, matrix_element=matrix_element)
        traces.extend(seg)

    fig = go.Figure(data=traces)
    fig.update_layout(
        xaxis_title='Flux (Φ_ext / Φ₀)',
        yaxis_title='Transition Frequency (GHz)',
        template='plotly_white',
        legend=dict(x=1.02, y=1)
    )
    return fig

if __name__ == '__main__':
    app.run(debug=True)
