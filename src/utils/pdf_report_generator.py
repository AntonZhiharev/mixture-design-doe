"""
Comprehensive PDF Report Generator for ANOVA Analysis
=====================================================
Generates professional PDF reports with all tables and graphics
"""

import io
import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from datetime import datetime


def generate_comprehensive_pdf_report(
    config,
    detailed_anova_df,
    model_summary,
    model_results,
    significant_terms,
    sig_terms_df,
    variance_display_df,
    half_normal_fig,
    residuals_fig,
    histogram_fig,
    predicted_fig,
    initial_data_df=None
):
    """
    Generate a comprehensive PDF report with all analysis data and graphics
    
    Parameters:
    -----------
    config : dict
        Analysis configuration
    detailed_anova_df : DataFrame
        Detailed ANOVA table
    model_summary : dict
        Model summary statistics
    model_results : dict
        Model performance results
    significant_terms : list
        List of significant term names
    sig_terms_df : DataFrame
        Significant terms with coefficients
    variance_display_df : DataFrame
        Variance analysis table
    half_normal_fig : plotly Figure
        Half-normal plot
    residuals_fig : plotly Figure
        Residuals vs fitted plot
    histogram_fig : plotly Figure
        Residuals histogram
    predicted_fig : plotly Figure
        Actual vs predicted plot
    initial_data_df : DataFrame, optional
        Initial experimental data with design matrix and response
    
    Returns:
    --------
    bytes : PDF file content
    """
    
    # Create PDF in memory
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=1.5*cm,
        leftMargin=1.5*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )
    
    # Container for PDF elements
    story = []
    
    # Styles
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=22,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#ff7f0e'),
        spaceAfter=12,
        spaceBefore=16,
        fontName='Helvetica-Bold'
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading3'],
        fontSize=13,
        textColor=colors.HexColor('#2ca02c'),
        spaceAfter=8,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    # ========== PAGE 1: TITLE & SUMMARY ==========
    story.append(Paragraph("COMPREHENSIVE ANOVA ANALYSIS REPORT", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 0.5*cm))
    
    # Analysis Information
    story.append(Paragraph("1. Analysis Configuration", heading1_style))
    
    info_data = [
        ['Parameter', 'Value'],
        ['Analysis Type', config['analysis_type']],
        ['Model Type', config.get('model_type', 'N/A').title()],
        ['Response Variable', config['response_col']],
        ['Number of Observations', str(len(model_results['residuals']))],
        ['Number of Factors', str(len(config['factor_cols']))],
        ['Factor Names', ', '.join(config['factor_cols'][:5]) + ('...' if len(config['factor_cols']) > 5 else '')]
    ]
    
    info_table = Table(info_data, colWidths=[8*cm, 9*cm])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.5*cm))
    
    # Model Performance Summary
    story.append(Paragraph("2. Model Performance Summary", heading1_style))
    
    perf_data = [
        ['Metric', 'Value', 'Interpretation'],
        ['R² Score', f"{model_summary['R_squared']:.4f}", 'Excellent' if model_summary['R_squared'] > 0.9 else 'Good' if model_summary['R_squared'] > 0.7 else 'Fair'],
        ['Adjusted R²', f"{model_summary['R_squared_adj']:.4f}", 'Accounts for model complexity'],
        ['RMSE', f"{model_results['rmse']:.4f}", 'Root Mean Square Error'],
        ['Residual MS', f"{model_summary['MS_error']:.4f}", 'Mean square error'],
        ['Error DF', str(int(model_summary['df_error'])), 'Degrees of freedom'],
        ['Total Parameters', str(model_summary['n_parameters']), 'Model complexity'],
        ['Sample Size', str(model_summary['n_samples']), 'Number of observations']
    ]
    
    perf_table = Table(perf_data, colWidths=[5*cm, 5*cm, 7*cm])
    perf_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(perf_table)
    story.append(Spacer(1, 0.5*cm))
    
    # Significant Terms Summary
    if significant_terms and len(significant_terms) > 0:
        story.append(Paragraph(f"3. Significant Terms ({len(significant_terms)} found at α=0.05)", heading1_style))
        
        sig_text = "<br/>".join([f"• <b>{term}</b>" for term in significant_terms[:15]])
        if len(significant_terms) > 15:
            sig_text += f"<br/>• <i>... and {len(significant_terms) - 15} more terms</i>"
        
        story.append(Paragraph(sig_text, styles['Normal']))
        story.append(Spacer(1, 0.3*cm))
    
    # ========== INITIAL EXPERIMENTAL DATA ==========
    if initial_data_df is not None and len(initial_data_df) > 0:
        story.append(PageBreak())
        story.append(Paragraph("4. Initial Experimental Data", heading1_style))
        story.append(Paragraph("Design matrix and response values used for analysis", styles['Normal']))
        story.append(Spacer(1, 0.3*cm))
        
        # Prepare data table - limit columns for PDF width
        data_rows = []
        columns_to_show = list(initial_data_df.columns)
        
        # If too many columns, we'll split into multiple tables
        max_cols_per_table = 8
        
        for col_start in range(0, len(columns_to_show), max_cols_per_table):
            col_end = min(col_start + max_cols_per_table, len(columns_to_show))
            current_cols = columns_to_show[col_start:col_end]
            
            # Build header
            header_row = current_cols
            
            # Build data rows (show ALL rows)
            table_data = [header_row]
            
            for idx in range(len(initial_data_df)):
                row_data = []
                for col in current_cols:
                    val = initial_data_df.iloc[idx][col]
                    # Format numbers, keep strings as-is
                    if isinstance(val, (int, np.integer)):
                        row_data.append(str(int(val)))
                    elif isinstance(val, (float, np.floating)):
                        row_data.append(f"{val:.3f}")
                    else:
                        row_data.append(str(val)[:15])  # Truncate long strings
                table_data.append(row_data)
            
            # Calculate column widths dynamically
            available_width = 17*cm
            col_width = available_width / len(current_cols)
            col_widths = [col_width] * len(current_cols)
            
            # Create table
            data_table = Table(table_data, colWidths=col_widths)
            data_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E75B6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 7),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ]))
            
            story.append(data_table)
            story.append(Spacer(1, 0.3*cm))
            
            # Add continuation note if columns split
            if col_end < len(columns_to_show):
                story.append(Paragraph(f"<i>Continued on next page (columns {col_start+1}-{col_end} of {len(columns_to_show)})...</i>", styles['Italic']))
                story.append(PageBreak())
        
        # Show total rows included
        story.append(Paragraph(f"<i>Complete data table: {len(initial_data_df)} observations shown</i>", styles['Italic']))
        
        story.append(Spacer(1, 0.5*cm))
    
    # ========== PAGE: DETAILED ANOVA TABLE ==========
    story.append(PageBreak())
    story.append(Paragraph("5. Detailed ANOVA Table", heading1_style))
    story.append(Paragraph("Complete analysis of variance for all model terms", styles['Normal']))
    story.append(Spacer(1, 0.3*cm))
    
    # Check if Pseudo Statistics are available
    has_pseudo_stats = 'Pseudo_t_Ratio' in detailed_anova_df.columns
    
    # Prepare ANOVA data (split into multiple tables if needed)
    anova_rows = []
    for idx, row in detailed_anova_df.iterrows():
        # Format F-statistic with scientific notation for large values
        f_stat_str = ""
        if not pd.isna(row['F_statistic']):
            if abs(row['F_statistic']) > 1000:
                f_stat_str = f"{row['F_statistic']:.2e}"
            else:
                f_stat_str = f"{row['F_statistic']:.2f}"
        
        row_data = [
            str(row['Source'])[:30],
            f"{row['Sum_of_Squares']:.3f}" if not pd.isna(row['Sum_of_Squares']) else "",
            f"{int(row['DF'])}" if not pd.isna(row['DF']) and row['DF'] >= 0 else "",
            f"{row['Mean_Square']:.3f}" if not pd.isna(row['Mean_Square']) else "",
            f_stat_str,
            f"{row['P_Value']:.2e}" if not pd.isna(row['P_Value']) else ""
        ]
        
        # Add Pseudo Statistics if available
        if has_pseudo_stats:
            row_data.extend([
                f"{row['Pseudo_t_Ratio']:.2f}" if not pd.isna(row['Pseudo_t_Ratio']) else "",
                f"{row['Pseudo_P_Value']:.2e}" if not pd.isna(row['Pseudo_P_Value']) else ""
            ])
        
        # Add Coefficient at the end
        row_data.append(f"{row['Coefficient']:.3f}" if not pd.isna(row['Coefficient']) else "")
        
        anova_rows.append(row_data)
    
    # Split into chunks of 25 rows
    chunk_size = 25
    for i in range(0, len(anova_rows), chunk_size):
        chunk = anova_rows[i:i+chunk_size]
        
        # Build header based on whether Pseudo Statistics are available
        if has_pseudo_stats:
            header = ['Source', 'SS', 'DF', 'MS', 'F', 'P-Val', 'Ps-t', 'Ps-P', 'Coef']
            col_widths = [3.5*cm, 1.5*cm, 0.8*cm, 1.5*cm, 1.5*cm, 1.5*cm, 1.2*cm, 1.5*cm, 1.5*cm]
        else:
            header = ['Source', 'SS', 'DF', 'MS', 'F', 'P-Value', 'Coef']
            col_widths = [4.5*cm, 2*cm, 1*cm, 2*cm, 1.5*cm, 2*cm, 2*cm]
        
        anova_data = [header] + chunk
        
        anova_table = Table(anova_data, colWidths=col_widths)
        anova_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#70AD47')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
       ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        story.append(anova_table)
        
        if i + chunk_size < len(anova_rows):
            story.append(Spacer(1, 0.3*cm))
            story.append(Paragraph(f"<i>Continued... (showing rows {i+1}-{min(i+chunk_size, len(anova_rows))} of {len(anova_rows)})</i>", styles['Italic']))
            story.append(PageBreak())
    
    # ========== VARIANCE ANALYSIS ==========
    if variance_display_df is not None and len(variance_display_df) > 0:
        story.append(PageBreak())
        story.append(Paragraph("6. Variance Analysis for Effect Estimates", heading1_style))
        story.append(Paragraph("Standard errors and 95% confidence intervals for model coefficients", styles['Normal']))
        story.append(Spacer(1, 0.3*cm))
        
        var_data = [['Term', 'Coefficient', 'Std Error', 't-Stat', '95% CI Lower', '95% CI Upper', 'Sig']]
        for idx, row in variance_display_df.iterrows():
            # Format t-statistic with scientific notation for large values
            t_stat_str = ""
            if not pd.isna(row['t-Stat']):
                if abs(row['t-Stat']) > 1000:
                    t_stat_str = f"{row['t-Stat']:.2e}"
                else:
                    t_stat_str = f"{row['t-Stat']:.2f}"
            
            var_data.append([
                str(row['Term'])[:25],
                f"{row['Coefficient']:.4f}",
                f"{row['Std Error']:.4f}",
                t_stat_str,
                f"{row['95% CI Lower']:.4f}",
                f"{row['95% CI Upper']:.4f}",
                '✓' if row['Significant'] else '✗'
            ])
        
        var_table = Table(var_data, colWidths=[4*cm, 2*cm, 2*cm, 1.5*cm, 2.2*cm, 2.2*cm, 1*cm])
        var_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#5B9BD5')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        story.append(var_table)
        story.append(Spacer(1, 0.3*cm))
    
    # ========== GRAPHICS ==========
    try:
        # Export plotly figures to images
        story.append(PageBreak())
        story.append(Paragraph("7. Graphical Analysis", heading1_style))
        story.append(Spacer(1, 0.3*cm))
        
        # Half-Normal Plot
        if half_normal_fig is not None:
            story.append(Paragraph("7.1 Half-Normal Probability Plot", heading2_style))
            story.append(Paragraph("Identifies significant effects graphically", styles['Normal']))
            story.append(Spacer(1, 0.2*cm))
            
            img_bytes = half_normal_fig.to_image(format="png", width=700, height=500, scale=2)
            img = Image(io.BytesIO(img_bytes), width=17*cm, height=12*cm)
            story.append(img)
            story.append(Spacer(1, 0.3*cm))
        
        # Residuals vs Fitted
        if residuals_fig is not None:
            story.append(PageBreak())
            story.append(Paragraph("7.2 Residuals vs Fitted Values", heading2_style))
            story.append(Paragraph("Check for patterns in residuals", styles['Normal']))
            story.append(Spacer(1, 0.2*cm))
            
            img_bytes = residuals_fig.to_image(format="png", width=700, height=450, scale=2)
            img = Image(io.BytesIO(img_bytes), width=17*cm, height=11*cm)
            story.append(img)
            story.append(Spacer(1, 0.3*cm))
        
        # Residuals Histogram
        if histogram_fig is not None:
            story.append(Paragraph("7.3 Residuals Distribution", heading2_style))
            story.append(Paragraph("Check for normality of residuals", styles['Normal']))
            story.append(Spacer(1, 0.2*cm))
            
            img_bytes = histogram_fig.to_image(format="png", width=700, height=450, scale=2)
            img = Image(io.BytesIO(img_bytes), width=17*cm, height=11*cm)
            story.append(img)
            story.append(Spacer(1, 0.3*cm))
        
        # Actual vs Predicted
        if predicted_fig is not None:
            story.append(PageBreak())
            story.append(Paragraph("7.4 Actual vs Predicted Values", heading2_style))
            story.append(Paragraph("Model prediction accuracy", styles['Normal']))
            story.append(Spacer(1, 0.2*cm))
            
            img_bytes = predicted_fig.to_image(format="png", width=700, height=500, scale=2)
            img = Image(io.BytesIO(img_bytes), width=17*cm, height=12*cm)
            story.append(img)
            story.append(Spacer(1, 0.3*cm))
    
    except Exception as e:
        story.append(Paragraph(f"<i>Note: Graphics could not be included. Install kaleido: pip install kaleido</i>", styles['Italic']))
        story.append(Paragraph(f"<i>Error: {str(e)}</i>", styles['Italic']))
    
    # ========== FINAL PAGE: MODEL FORMULA ==========
    if sig_terms_df is not None and len(sig_terms_df) > 0:
        story.append(PageBreak())
        story.append(Paragraph("8. Final Model Formula", heading1_style))
        story.append(Paragraph(f"Fitted model with {len(sig_terms_df)} significant terms", styles['Normal']))
        story.append(Spacer(1, 0.3*cm))
        
        # Build formula
        formula_parts = []
        for idx, row in sig_terms_df.iterrows():
            term = row['Term']
            coef = row['Coefficient']
            if coef >= 0:
                sign = "+ " if len(formula_parts) > 0 else ""
                formula_parts.append(f"{sign}{abs(coef):.4f} × {term}")
            else:
                formula_parts.append(f"- {abs(coef):.4f} × {term}")
        
        formula_text = " ".join(formula_parts)
        
        # Split into multiple lines if too long
        max_line_length = 80
        formula_lines = []
        current_line = "Response = "
        
        for part in formula_parts:
            if len(current_line) + len(part) > max_line_length:
                formula_lines.append(current_line)
                current_line = "           " + part  # Indent continuation
            else:
                current_line += part
        formula_lines.append(current_line)
        
        for line in formula_lines:
            story.append(Paragraph(f"<font name='Courier' size=9>{line}</font>", styles['Code']))
        
        story.append(Spacer(1, 0.5*cm))
        
        # Summary metrics
        summary_data = [
            ['Metric', 'Value'],
            ['Terms in Final Model', str(len(sig_terms_df))],
            ['R² Score', f"{model_summary['R_squared']:.4f}"],
            ['Adjusted R²', f"{model_summary['R_squared_adj']:.4f}"],
            ['RMSE', f"{model_results['rmse']:.4f}"]
        ]
        
        summary_table = Table(summary_data, colWidths=[8*cm, 6*cm])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ED7D31')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(summary_table)
    
    # Footer
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph("<i>End of Report</i>", styles['Italic']))
    story.append(Paragraph(f"<i>Generated by DOE Analysis System - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>", styles['Italic']))
    
    # Build PDF
    doc.build(story)
    
    return pdf_buffer.getvalue()
