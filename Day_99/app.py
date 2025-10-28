from flask import Flask, render_template, request, jsonify
import pandas as pd
import json
from data_warehouse_etl import DataWarehouse
from olap_analysis import OLAPAnalyzer

app = Flask(__name__)

# Initialize Data Warehouse
dw = DataWarehouse()
dw.generate_sample_data()
analyzer = OLAPAnalyzer(dw)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/olap_operation', methods=['POST'])
def olap_operation():
    operation = request.json.get('operation')
    
    try:
        if operation == 'roll_up':
            result = dw.olap_roll_up('time', ['total_amount', 'quantity'])
            result_data = result.reset_index().to_dict('records')
            
        elif operation == 'drill_down':
            result = dw.olap_drill_down('product', ['total_amount', 'quantity'])
            result_data = result.reset_index().to_dict('records')
            
        elif operation == 'slice':
            result = dw.olap_slice('region', 'North', ['total_amount', 'quantity'])
            result_data = [{'measure': k, 'value': v} for k, v in result.items()]
            
        elif operation == 'dice':
            conditions = ["year == 2023", "quarter == 1", "category == 'Electronics'"]
            result = dw.olap_dice(conditions, ['total_amount', 'quantity'])
            result_data = [{'measure': k, 'value': v} for k, v in result.items()]
            
        else:
            return jsonify({'error': 'Invalid operation'})
        
        return jsonify({
            'success': True,
            'operation': operation,
            'data': result_data
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/get_analysis')
def get_analysis():
    """Get pre-calculated analysis data"""
    try:
        # Sales trend data
        sales_trend = analyzer.create_sales_trend_analysis()
        trend_data = sales_trend.to_dict('records')
        
        # Product performance data
        product_perf = analyzer.create_product_performance_matrix()
        product_data = product_perf.head(10).to_dict('records')
        
        return jsonify({
            'sales_trend': trend_data,
            'product_performance': product_data
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/schema_info')
def schema_info():
    """Get data warehouse schema information"""
    schema_info = {
        'fact_tables': ['sales_fact'],
        'dimension_tables': list(dw.dimension_tables.keys()),
        'measures': ['total_amount', 'quantity', 'unit_price'],
        'dimensions': ['time', 'product', 'customer', 'store']
    }
    return jsonify(schema_info)

if __name__ == '__main__':
    app.run(debug=True)