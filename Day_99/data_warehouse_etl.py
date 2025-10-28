import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

class DataWarehouse:
    def __init__(self):
        self.fact_table = None
        self.dimension_tables = {}
        
    def generate_sample_data(self):
        """Generate sample data for data warehouse simulation"""
        np.random.seed(42)
        
        # Date Dimension
        dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
        date_dim = pd.DataFrame({
            'date_id': range(1, len(dates) + 1),
            'full_date': dates,
            'day': dates.day,
            'month': dates.month,
            'quarter': dates.quarter,
            'year': dates.year,
            'day_of_week': dates.dayofweek,
            'is_weekend': dates.dayofweek.isin([5, 6]).astype(int)
        })
        
        # Product Dimension
        products = ['Laptop', 'Smartphone', 'Tablet', 'Monitor', 'Keyboard', 
                   'Mouse', 'Headphones', 'Printer', 'Scanner', 'Webcam']
        product_dim = pd.DataFrame({
            'product_id': range(1, len(products) + 1),
            'product_name': products,
            'category': ['Electronics'] * 10,
            'subcategory': ['Computers', 'Mobile', 'Mobile', 'Computers', 
                           'Accessories', 'Accessories', 'Audio', 'Office', 
                           'Office', 'Accessories'],
            'price': [1200, 800, 400, 300, 50, 25, 150, 200, 180, 80]
        })
        
        # Customer Dimension
        customer_dim = pd.DataFrame({
            'customer_id': range(1, 101),
            'customer_name': [f'Customer_{i}' for i in range(1, 101)],
            'region': np.random.choice(['North', 'South', 'East', 'West'], 100),
            'country': np.random.choice(['USA', 'Canada', 'UK', 'Germany'], 100),
            'customer_segment': np.random.choice(['Premium', 'Standard', 'Basic'], 100)
        })
        
        # Store Dimension
        store_dim = pd.DataFrame({
            'store_id': range(1, 11),
            'store_name': [f'Store_{i}' for i in range(1, 11)],
            'location': np.random.choice(['Mall', 'Downtown', 'Suburban'], 10),
            'size_sqft': np.random.randint(1000, 5000, 10)
        })
        
        # Fact Table - Sales
        n_transactions = 5000
        fact_sales = pd.DataFrame({
            'sale_id': range(1, n_transactions + 1),
            'date_id': np.random.choice(date_dim['date_id'], n_transactions),
            'product_id': np.random.choice(product_dim['product_id'], n_transactions),
            'customer_id': np.random.choice(customer_dim['customer_id'], n_transactions),
            'store_id': np.random.choice(store_dim['store_id'], n_transactions),
            'quantity': np.random.randint(1, 5, n_transactions),
            'unit_price': np.random.uniform(10, 1500, n_transactions),
            'discount': np.random.uniform(0, 0.3, n_transactions)
        })
        
        # Calculate total amount
        fact_sales['total_amount'] = fact_sales['quantity'] * fact_sales['unit_price'] * (1 - fact_sales['discount'])
        
        self.dimension_tables = {
            'date_dim': date_dim,
            'product_dim': product_dim,
            'customer_dim': customer_dim,
            'store_dim': store_dim
        }
        self.fact_table = fact_sales
        
        print("✅ Data Warehouse tables created successfully!")
        return self.fact_table, self.dimension_tables
    
    def star_schema_query(self, dimensions, measures):
        """Perform OLAP query on star schema"""
        # Merge fact table with dimension tables
        merged_data = self.fact_table.copy()
        
        if 'date' in dimensions:
            merged_data = merged_data.merge(
                self.dimension_tables['date_dim'], 
                on='date_id', 
                how='left'
            )
        
        if 'product' in dimensions:
            merged_data = merged_data.merge(
                self.dimension_tables['product_dim'], 
                on='product_id', 
                how='left'
            )
        
        if 'customer' in dimensions:
            merged_data = merged_data.merge(
                self.dimension_tables['customer_dim'], 
                on='customer_id', 
                how='left'
            )
        
        if 'store' in dimensions:
            merged_data = merged_data.merge(
                self.dimension_tables['store_dim'], 
                on='store_id', 
                how='left'
            )
        
        # Group by dimensions and calculate measures
        result = merged_data.groupby(dimensions)[measures].agg(['sum', 'mean', 'count']).round(2)
        return result
    
    def olap_roll_up(self, dimension_hierarchy, measures):
        """OLAP Roll-up operation - moving up in hierarchy"""
        print(f"🔝 Roll-up operation: {dimension_hierarchy}")
        
        merged_data = self.fact_table.merge(
            self.dimension_tables['date_dim'], on='date_id', how='left'
        )
        
        if dimension_hierarchy == 'time':
            # Roll-up from day -> month -> quarter -> year
            result = merged_data.groupby(['year', 'quarter', 'month'])[measures].sum().round(2)
        
        return result
    
    def olap_drill_down(self, dimension_hierarchy, measures):
        """OLAP Drill-down operation - moving down in hierarchy"""
        print(f"🔍 Drill-down operation: {dimension_hierarchy}")
        
        merged_data = self.fact_table.merge(
            self.dimension_tables['product_dim'], on='product_id', how='left'
        )
        
        if dimension_hierarchy == 'product':
            # Drill-down from category -> subcategory -> product
            result = merged_data.groupby(['category', 'subcategory', 'product_name'])[measures].sum().round(2)
        
        return result
    
    def olap_slice(self, dimension, value, measures):
        """OLAP Slice operation - selecting specific dimension value"""
        print(f"🔪 Slice operation: {dimension} = {value}")
        
        if dimension == 'region':
            merged_data = self.fact_table.merge(
                self.dimension_tables['customer_dim'], on='customer_id', how='left'
            )
            sliced_data = merged_data[merged_data['region'] == value]
            result = sliced_data[measures].sum().round(2)
        
        return result
    
    def olap_dice(self, conditions, measures):
        """OLAP Dice operation - selecting specific cells from multiple dimensions"""
        print(f"🎲 Dice operation with conditions")
        
        merged_data = self.fact_table.merge(
            self.dimension_tables['date_dim'], on='date_id', how='left'
        ).merge(
            self.dimension_tables['product_dim'], on='product_id', how='left'
        ).merge(
            self.dimension_tables['customer_dim'], on='customer_id', how='left'
        )
        
        # Apply conditions
        for condition in conditions:
            merged_data = merged_data.query(condition)
        
        result = merged_data[measures].sum().round(2)
        return result
    
    def save_data(self):
        """Save data warehouse tables to files"""
        self.fact_table.to_csv('fact_sales.csv', index=False)
        for name, table in self.dimension_tables.items():
            table.to_csv(f'{name}.csv', index=False)
        print("✅ Data saved to CSV files")

def main():
    print("🚀 Day 99: Data Warehouse & OLAP Systems")
    print("=" * 50)
    
    # Initialize data warehouse
    dw = DataWarehouse()
    
    # Generate sample data
    print("📊 Generating data warehouse tables...")
    fact_table, dimension_tables = dw.generate_sample_data()
    
    print(f"Fact table shape: {fact_table.shape}")
    for name, table in dimension_tables.items():
        print(f"{name} shape: {table.shape}")
    
    # Perform OLAP operations
    print("\n🎯 Performing OLAP Operations:")
    
    # Roll-up example
    print("\n1. Roll-up (Time Hierarchy):")
    rollup_result = dw.olap_roll_up('time', ['total_amount', 'quantity'])
    print(rollup_result.head())
    
    # Drill-down example
    print("\n2. Drill-down (Product Hierarchy):")
    drilldown_result = dw.olap_drill_down('product', ['total_amount', 'quantity'])
    print(drilldown_result.head())
    
    # Slice example
    print("\n3. Slice (Region = 'North'):")
    slice_result = dw.olap_slice('region', 'North', ['total_amount', 'quantity'])
    print(slice_result)
    
    # Dice example
    print("\n4. Dice (Multiple Conditions):")
    dice_conditions = ["year == 2023", "quarter == 1", "category == 'Electronics'"]
    dice_result = dw.olap_dice(dice_conditions, ['total_amount', 'quantity'])
    print(dice_result)
    
    # Star schema query
    print("\n5. Star Schema Query:")
    star_result = dw.star_schema_query(
        dimensions=['year', 'quarter', 'region'],
        measures=['total_amount', 'quantity']
    )
    print(star_result.head())
    
    # Save data
    dw.save_data()
    
    print(f"\n✅ Data Warehouse simulation completed!")
    print(f"   Total sales transactions: {len(fact_table)}")
    print(f"   Total revenue: ${fact_table['total_amount'].sum():,.2f}")

if __name__ == "__main__":
    main()