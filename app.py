import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import plotly.figure_factory as ff

# ==========================================
# 1. SYNTHETIC DATA SETUP (Swap with your actual dataset)
# ==========================================
np.random.seed(42)
n_students = 500

df = pd.DataFrame({
    'Student_ID': [f'STU-{i:04d}' for i in range(n_students)],
    'Semester_Average_Grade': np.random.uniform(1.5, 5.0, n_students),
    'Semester_Approved_Units': np.random.uniform(3, 15, n_students),
    'Semester_Credited_Units': np.random.uniform(0, 30, n_students),
    'Semester_Enrolled_Units': np.random.uniform(6, 18, n_students),
    'Semester_Evaluated_Units': np.random.uniform(6, 18, n_students),
    'Course_Chosen': np.random.choice(['Business', 'Engineering', 'Arts', 'Science'], n_students),
    'Age': np.random.uniform(18, 35, n_students),
    'Gender': np.random.choice(['Male', 'Female', 'Other'], n_students),
    'Marital_Status': np.random.choice(['Single', 'Married'], n_students),
    'Employment_Status': np.random.choice(['Unemployed', 'Part-Time', 'Full-Time'], n_students),
    'Parental_Income_Level': np.random.uniform(15000, 120000, n_students),
    'Residence_Location': np.random.choice(['Urban', 'Suburban', 'Rural'], n_students),
    'Parental_Education': np.random.choice(['High School', 'Bachelor', 'Master'], n_students),
    'Actual_Dropout': np.random.choice([0, 1], n_students, p=[0.75, 0.25]), # y_test
    'LR_Risk_Probability': np.random.uniform(0, 1, n_students) # y_pred_probs
})

# Apply the Pre-Trained Decision Tree Persona Logic
def assign_final_persona(row, threshold=0.3):
    flagged = 1 if row['LR_Risk_Probability'] >= threshold else 0
    if flagged == 0:
        return "Not At Risk (No Intervention Needed)"
        
    gpa = row['Semester_Average_Grade']
    income = row['Parental_Income_Level']
    approved = row['Semester_Approved_Units']
    credited = row['Semester_Credited_Units']
    age = row['Age']
    
    if gpa < 2.46 and income < 25030:
        return "⚠️ Financial & Academic Hardship"
    elif gpa >= 2.46 and approved < 6.5 and credited > 6.5:
        return "⚠️ Transfer & Part-Time Disconnect"
    elif gpa >= 2.46 and approved > 6.5 and age > 26:
        return "⚠️ Adult Learner Burnout"
    else:
        return "⚠️ General Attrition Risk"

df['Persona'] = df.apply(lambda row: assign_final_persona(row), axis=1)


# ==========================================
# 2. DASH APP INITIALIZATION & LAYOUT
# ==========================================
app = dash.Dash(__name__)

# CSS Styles for clean cards
card_style = {
    'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)', 'borderRadius': '5px',
    'padding': '20px', 'backgroundColor': '#f9f9f9', 'flex': '1', 'margin': '10px'
}
kpi_style = {
    'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)', 'borderRadius': '5px',
    'padding': '20px', 'textAlign': 'center', 'flex': '1', 'margin': '10px'
}

app.layout = html.Div(style={'fontFamily': 'Arial'}, children=[
    html.H1("University Student Retention System", style={'textAlign': 'center', 'color': '#2c3e50'}),
    
    dcc.Tabs([
        # --- TAB 1: STUDENT PROFILER (Advisor View) ---
        dcc.Tab(label='1. Student Profiler', children=[
            html.Div(style={'padding': '20px', 'maxWidth': '1000px', 'margin': 'auto'}, children=[
                html.Label("Search Student ID:", style={'fontWeight': 'bold', 'fontSize': '16px'}),
                dcc.Dropdown(
                    id='student-dropdown',
                    options=[{'label': i, 'value': i} for i in df['Student_ID']],
                    value=df['Student_ID'].iloc[0],
                    style={'marginBottom': '20px'}
                ),
                html.H2(id='persona-badge', style={'color': '#d35400', 'textAlign': 'center'}),
                
                html.Div(style={'display': 'flex', 'justifyContent': 'space-between'}, children=[
                    # Academic Card
                    html.Div(style=card_style, children=[
                        html.H3("📚 Academic Factors", style={'borderBottom': '2px solid #3498db'}),
                        html.Div(id='card-academic')
                    ]),
                    # Demographic Card
                    html.Div(style=card_style, children=[
                        html.H3("👤 Demographic Factors", style={'borderBottom': '2px solid #e74c3c'}),
                        html.Div(id='card-demographic')
                    ]),
                    # Socioeconomic Card
                    html.Div(style=card_style, children=[
                        html.H3("🏠 Socioeconomic Factors", style={'borderBottom': '2px solid #2ecc71'}),
                        html.Div(id='card-socioeconomic')
                    ])
                ])
            ])
        ]),
        
        # --- TAB 2: FINANCIAL ROI SIMULATOR (Executive View) ---
        dcc.Tab(label='2. Financial ROI Simulator', children=[
            html.Div(style={'display': 'flex', 'padding': '20px'}, children=[
                
                # LEFT PANEL: Controls
                html.Div(style={'flex': '1', 'padding': '20px', 'backgroundColor': '#f2f2f2', 'borderRadius': '5px'}, children=[
                    html.H3("⚙️ Optimization Engine"),
                    
                    # The Optimal Threshold Toggle
                    dcc.Checklist(
                        id='toggle-optimal',
                        options=[{'label': ' Lock at Optimal Threshold (0.30)', 'value': 'locked'}],
                        value=['locked'],
                        style={'fontWeight': 'bold', 'marginBottom': '10px', 'fontSize': '16px'}
                    ),
                    
                    html.Label("Manual Decision Threshold:"),
                    dcc.Slider(id='slider-threshold', min=0, max=1, step=0.01, value=0.30, 
                               marks={0: '0', 0.3: '0.3', 1: '1'}, disabled=True),
                    
                    html.Br(),
                    html.Label("Retention Success Rate ($x\%$):"),
                    dcc.Input(id='input-success-rate', type='number', value=30, style={'width': '100%', 'marginBottom': '15px'}),
                    
                    html.Label("Average Years Completed ($y$):"),
                    dcc.Input(id='input-years', type='number', value=1, style={'width': '100%', 'marginBottom': '15px'}),
                    
                    html.Label("Cost per Counseling Session ($):"),
                    dcc.Input(id='input-cost', type='number', value=500, style={'width': '100%', 'marginBottom': '15px'}),
                ]),
                
                # RIGHT PANEL: Output Metrics
                html.Div(style={'flex': '3', 'paddingLeft': '20px'}, children=[
                    # KPI Boxes
                    html.Div(style={'display': 'flex'}, children=[
                        html.Div(style={**kpi_style, 'backgroundColor': '#e8f8f5'}, children=[
                            html.H4("Revenue Saved"),
                            html.H2(id='kpi-revenue', style={'color': '#27ae60'})
                        ]),
                        html.Div(style={**kpi_style, 'backgroundColor': '#fdedec'}, children=[
                            html.H4("Intervention Cost"),
                            html.H2(id='kpi-cost', style={'color': '#c0392b'})
                        ]),
                        html.Div(style={**kpi_style, 'backgroundColor': '#ebf5fb'}, children=[
                            html.H4("Net Business ROI"),
                            html.H2(id='kpi-roi', style={'color': '#2980b9'})
                        ])
                    ]),
                    
                    # Confusion Matrix Plot
                    html.Div(dcc.Graph(id='plot-confusion-matrix'))
                ])
            ])
        ])
    ])
])


# ==========================================
# 3. CALLBACKS (Interactivity Logic)
# ==========================================

# Callback 1: Populate the Student Profiler
@app.callback(
    [Output('persona-badge', 'children'), Output('card-academic', 'children'),
     Output('card-demographic', 'children'), Output('card-socioeconomic', 'children')],
    [Input('student-dropdown', 'value')]
)
def update_profiler(student_id):
    student = df[df['Student_ID'] == student_id].iloc[0]
    
    badge = f"Intervention Profile: {student['Persona']}"
    
    academic = html.Ul([
        html.Li(f"Major: {student['Course_Chosen']}"),
        html.Li(f"GPA: {student['Semester_Average_Grade']:.2f}"),
        html.Li(f"Enrolled Units: {student['Semester_Enrolled_Units']:.1f}"),
        html.Li(f"Approved Units: {student['Semester_Approved_Units']:.1f}"),
        html.Li(f"Credited (Transfer) Units: {student['Semester_Credited_Units']:.1f}")
    ])
    demographic = html.Ul([
        html.Li(f"Age: {student['Age']:.1f}"),
        html.Li(f"Gender: {student['Gender']}"),
        html.Li(f"Marital Status: {student['Marital_Status']}"),
        html.Li(f"Employment: {student['Employment_Status']}")
    ])
    socioeconomic = html.Ul([
        html.Li(f"Parental Income: ${student['Parental_Income_Level']:,.2f}"),
        html.Li(f"Location: {student['Residence_Location']}"),
        html.Li(f"Parent Education: {student['Parental_Education']}")
    ])
    return badge, academic, demographic, socioeconomic


# Callback 2: Manage the Toggle and Slider state
@app.callback(
    [Output('slider-threshold', 'value'), Output('slider-threshold', 'disabled')],
    [Input('toggle-optimal', 'value')],
    [State('slider-threshold', 'value')]
)
def update_slider_state(toggle_val, current_slider_val):
    if toggle_val and 'locked' in toggle_val:
        return 0.30, True # Lock at 0.30 and disable slider
    return current_slider_val, False # Unlock


# Callback 3: Compute Financials and Confusion Matrix
@app.callback(
    [Output('kpi-revenue', 'children'), Output('kpi-cost', 'children'),
     Output('kpi-roi', 'children'), Output('plot-confusion-matrix', 'figure')],
    [Input('slider-threshold', 'value'), Input('input-success-rate', 'value'),
     Input('input-years', 'value'), Input('input-cost', 'value')]
)
def update_financials(threshold, success_rate, years, counseling_cost):
    # 1. Apply threshold to get custom predictions
    y_pred_custom = (df['LR_Risk_Probability'] >= threshold).astype(int)
    y_actual = df['Actual_Dropout']
    
    # 2. Extract Confusion Matrix metrics manually to avoid reshaping errors
    tp = sum((y_pred_custom == 1) & (y_actual == 1))
    fp = sum((y_pred_custom == 1) & (y_actual == 0))
    tn = sum((y_pred_custom == 0) & (y_actual == 0))
    fn = sum((y_pred_custom == 0) & (y_actual == 1))
    
    # 3. Apply your exact financial logic
    students_retained = tp * (success_rate / 100)
    revenue_loss_per_student = 40000 - (10000 * years)
    total_revenue_saved = students_retained * revenue_loss_per_student
    
    intervention_cost = (tp + fp) * counseling_cost
    net_roi = total_revenue_saved - intervention_cost
    
    # 4. Format KPIs
    str_rev = f"${total_revenue_saved:,.0f}"
    str_cost = f"${intervention_cost:,.0f}"
    str_roi = f"${net_roi:,.0f}"
    
    # 5. Build clean Confusion Matrix Heatmap
    z = [[tn, fp], [fn, tp]]
    x = ['Predicted Retained (0)', 'Predicted Dropout (1)']
    y = ['Actual Retained (0)', 'Actual Dropout (1)']
    
    fig = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Blues', showscale=True)
    fig.update_layout(title_text='Live Logistic Regression Confusion Matrix', title_x=0.5)
    
    return str_rev, str_cost, str_roi, fig


if __name__ == '__main__':
    app.run_server(debug=True)