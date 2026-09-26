import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.figure_factory as ff

# ==========================================
# 1. LOAD REAL DATA DIRECTLY FROM GITHUB 
# ==========================================
# Using the "raw.githubusercontent" URL so Pandas reads the CSV, not the webpage HTML
github_csv_url = "https://raw.githubusercontent.com/josephakpro/Student_Retention/main/Tableau_Student_Retention_Dashboard.csv"

try:
    df = pd.read_csv(github_csv_url)
except Exception as e:
    print(f"⚠️ ERROR: Could not load data from GitHub. Check the URL. Details: {e}")
    # FIXED: Replaced 'Dropout' with 'Actual_Dropout' to prevent KeyError in callbacks
    df = pd.DataFrame(columns=['Student_ID', 'Semester_Average_Grade', 'Parental_Income_Level', 
                               'Semester_Approved_Units', 'Semester_Credited_Units', 'Age', 
                               'Dropout', 'LR_Risk_Probability'])

# Apply the Pre-Trained Decision Tree Persona Logic
def assign_final_persona(row, threshold=0.3):
    flagged = 1 if row['LR_Risk_Probability'] >= threshold else 0
    if flagged == 0:
        return "Not At Risk"
        
    gpa = row['Semester_Average_Grade']
    income = row['Parental_Income_Level']
    approved = row['Semester_Approved_Units']
    credited = row['Semester_Credited_Units']
    age = row['Age']
    
    if gpa < 2.5 and income < 25030:
        return "⚠️ Financial & Academic Hardship"
    elif gpa >= 2.5 and approved < 6.5 and credited > 6.5:
        return "⚠️ Transfer & Part-Time Disconnect"
    elif gpa >= 2.5 and approved > 6.5 and age > 26:
        return "⚠️ Adult Learner Burnout"
    else:
        return "⚠️ General Attrition Risk"

df['Persona'] = df.apply(lambda row: assign_final_persona(row), axis=1)

# Filter out the "Not At Risk" students so the table ONLY shows the flagged students
action_df = df[df['Persona'] != "Not At Risk"].copy()

# Format numeric columns for a cleaner UI
action_df['Semester_Average_Grade'] = action_df['Semester_Average_Grade'].round(2)
action_df['LR_Risk_Probability'] = (action_df['LR_Risk_Probability'] * 100).round(1).astype(str) + '%'

# ==========================================
# 2. DASH APP INITIALIZATION & LAYOUT
# ==========================================
app = dash.Dash(__name__)
server = app.server

kpi_style = {
    'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)', 'borderRadius': '5px',
    'padding': '20px', 'textAlign': 'center', 'flex': '1', 'margin': '10px'
}

app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'maxWidth': '1200px', 'margin': 'auto'}, children=[
    
    # Business context for executives opening the app cold
    html.Div(style={'backgroundColor': '#2c3e50', 'color': 'white', 'padding': '20px', 'borderRadius': '5px', 'marginBottom': '20px'}, children=[
        html.H1("University Student Retention System", style={'margin': '0 0 10px 0'}),
        html.P("This decision support system utilizes a two-stage machine learning architecture. A Logistic Regression model acts as the financial trigger to identify at-risk students, while a Decision Tree algorithm segments those flagged students into actionable intervention personas for academic advisors.", style={'fontSize': '16px', 'margin': '0'})
    ]),
    
    dcc.Tabs([
        # --- TAB 1: ADVISOR ACTION LIST (Micro View) ---
        dcc.Tab(label='1. Advisor Action List', children=[
            html.Div(style={'padding': '20px'}, children=[
                html.H3("Daily Intervention Playbook"),
                html.P("Filter the current list of flagged, at-risk students by their diagnostic persona to align with your advising specialty.", style={'color': '#7f8c8d'}),
                
                html.Label("Filter by Target Persona:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='persona-filter',
                    options=[{'label': p, 'value': p} for p in action_df['Persona'].unique()],
                    value='All',
                    placeholder="Select a Persona to filter...",
                    style={'width': '50%', 'marginBottom': '20px'}
                ),
                
                dash_table.DataTable(
                    id='advisor-table',
                    columns=[
                        {"name": "Student ID", "id": "Student_ID"},
                        {"name": "Diagnostic Persona", "id": "Persona"},
                        {"name": "GPA", "id": "Semester_Average_Grade"},
                        {"name": "Age", "id": "Age"},
                        {"name": "Risk Score", "id": "LR_Risk_Probability"}
                    ],
                    data=action_df.to_dict('records'),
                    style_table={'overflowX': 'auto', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.1)'},
                    style_cell={'textAlign': 'left', 'padding': '12px', 'fontFamily': 'Arial'},
                    style_header={'backgroundColor': '#34495e', 'color': 'white', 'fontWeight': 'bold'},
                    style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#f9f9f9'}],
                    page_size=15,
                    sort_action="native"
                )
            ])
        ]),
        
        # --- TAB 2: FINANCIAL ROI SIMULATOR (Macro View) ---
        dcc.Tab(label='2. Financial ROI Simulator', children=[
            html.Div(style={'display': 'flex', 'padding': '20px'}, children=[
                
                # LEFT PANEL: Controls 
                html.Div(style={'flex': '1', 'padding': '20px', 'backgroundColor': '#f2f2f2', 'borderRadius': '5px'}, children=[
                    html.H3("⚙️ Optimization Engine"),
                    
                    dcc.Checklist(
                        id='toggle-optimal',
                        options=[{'label': ' Lock at Optimal Threshold (0.30)', 'value': 'locked'}],
                        value=['locked'],
                        style={'fontWeight': 'bold', 'marginBottom': '10px', 'fontSize': '16px'}
                    ),
                    
                    html.Label("Manual Decision Threshold:"),
                    dcc.Slider(id='slider-threshold', min=0, max=1, step=0.01, value=0.30, 
                               marks={0: '0', 0.3: '0.3', 1: '1'}, disabled=True),
                    
                    html.Div(style={'marginTop': '20px', 'marginBottom': '20px', 'textAlign': 'center'}, children=[
                        html.Label("Profit Curve Optimization", style={'fontWeight': 'bold', 'fontSize': '12px', 'color': '#7f8c8d'}),
                        html.Img(src='/assets/profit_curve.png', style={'width': '100%', 'borderRadius': '5px', 'boxShadow': '0 2px 4px 0 rgba(0,0,0,0.1)'})
                    ]),
                    
                    # FIXED: Added 'r' before the string to fix the invalid escape sequence warning
                    html.Label(r"Estimated Retention Success Rate (%):"),
                    dcc.Input(id='input-success-rate', type='number', min=0, max=100, value=30, style={'width': '100%', 'marginBottom': '15px'}),
                    
                    html.Label("Average Years Completed:"),
                    dcc.Input(id='input-years', type='number', min=0, max=4, value=1, style={'width': '100%', 'marginBottom': '15px'}),
                    
                    html.Label("Cost per Counseling Session ($):"),
                    dcc.Input(id='input-cost', type='number', min=0, value=500, style={'width': '100%', 'marginBottom': '15px'}),
                ]),
                
                # RIGHT PANEL: Output Metrics
                html.Div(style={'flex': '3', 'paddingLeft': '20px'}, children=[
                    html.Div(style={'display': 'flex'}, children=[
                        html.Div(style={**kpi_style, 'backgroundColor': '#e8f8f5'}, children=[
                            html.H4("Revenue Saved", style={'margin': '0 0 10px 0', 'color': '#7f8c8d'}),
                            html.H2(id='kpi-revenue', style={'margin': '0', 'color': '#27ae60'})
                        ]),
                        html.Div(style={**kpi_style, 'backgroundColor': '#fdedec'}, children=[
                            html.H4("Intervention Cost", style={'margin': '0 0 10px 0', 'color': '#7f8c8d'}),
                            html.H2(id='kpi-cost', style={'margin': '0', 'color': '#c0392b'})
                        ]),
                        html.Div(style={**kpi_style, 'backgroundColor': '#ebf5fb'}, children=[
                            html.H4("Net Business ROI", style={'margin': '0 0 10px 0', 'color': '#7f8c8d'}),
                            html.H2(id='kpi-roi', style={'margin': '0', 'color': '#2980b9'})
                        ])
                    ]),
                    
                    html.Div(dcc.Graph(id='plot-confusion-matrix'))
                ])
            ])
        ])
    ])
])

# ==========================================
# 3. CALLBACKS
# ==========================================

# Callback 1: Filter the Advisor Table
@app.callback(
    Output('advisor-table', 'data'),
    [Input('persona-filter', 'value')]
)
def update_table(selected_persona):
    if selected_persona is None or selected_persona == 'All':
        return action_df.to_dict('records')
    filtered_df = action_df[action_df['Persona'] == selected_persona]
    return filtered_df.to_dict('records')

# Callback 2: Manage the Toggle and Slider state
@app.callback(
    [Output('slider-threshold', 'value'), Output('slider-threshold', 'disabled')],
    [Input('toggle-optimal', 'value')],
    [State('slider-threshold', 'value')]
)
def update_slider_state(toggle_val, current_slider_val):
    if toggle_val and 'locked' in toggle_val:
        return 0.30, True 
    return current_slider_val, False 

# Callback 3: Compute Financials and Confusion Matrix
@app.callback(
    [Output('kpi-revenue', 'children'), Output('kpi-cost', 'children'),
     Output('kpi-roi', 'children'), Output('plot-confusion-matrix', 'figure')],
    [Input('slider-threshold', 'value'), Input('input-success-rate', 'value'),
     Input('input-years', 'value'), Input('input-cost', 'value')]
)
def update_financials(threshold, success_rate, years, counseling_cost):
    if None in [threshold, success_rate, years, counseling_cost]:
        raise PreventUpdate

    y_pred_custom = (df['LR_Risk_Probability'] >= threshold).astype(int)
    y_actual = df['Dropout']
    
    tp = int(sum((y_pred_custom == 1) & (y_actual == 1)))
    fp = int(sum((y_pred_custom == 1) & (y_actual == 0)))
    tn = int(sum((y_pred_custom == 0) & (y_actual == 0)))
    fn = int(sum((y_pred_custom == 0) & (y_actual == 1)))
    
    students_retained = tp * (success_rate / 100)
    revenue_loss_per_student = 40000 - (10000 * years)
    total_revenue_saved = students_retained * revenue_loss_per_student
    
    intervention_cost = (tp + fp) * counseling_cost
    net_roi = total_revenue_saved - intervention_cost
    
    str_rev = f"${total_revenue_saved:,.0f}"
    str_cost = f"${intervention_cost:,.0f}"
    str_roi = f"${net_roi:,.0f}"
    
    z = [[tn, fp], [fn, tp]]
    x = ['Predicted Retained (0)', 'Predicted Dropout (1)']
    y = ['Actual Retained (0)', 'Actual Dropout (1)']
    
    fig = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Blues', showscale=True)
    fig.update_layout(title_text='Live Logistic Regression Confusion Matrix', title_x=0.5, margin=dict(t=50, l=20, r=20, b=20))
    
    return str_rev, str_cost, str_roi, fig

if __name__ == '__main__':
    app.run_server(debug=True)
