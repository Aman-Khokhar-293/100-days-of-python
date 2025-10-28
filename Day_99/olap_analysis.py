import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_warehouse_etl import DataWarehouse

class OLAPAnalyzer:
    def __init__(self, data_warehouse):
        self.dw = data_warehouse
        
    def create_sales_trend_analysis(self):
        """Analyze sales trends over time"""
        merged_data = self.dw.fact_table.merge(
            self.dw.dimension_tables['date_dim'], on='date_id', how='left'
        )
        
        monthly_sales = merged_data.groupby(['year', 'month'])['total_amount'].sum().reset_index()
        monthly_sales['period'] = monthly_sales['year'].astype(str) + '-' + monthly_sales['month'].astype(str).str.zfill(2)
        
        plt.figure(figsize=(12, 6))
        plt.plot(monthly_sales['period'], monthly_sales['total_amount'], marker='o')
        plt.title('Monthly Sales Trend')
        plt.xlabel('Month')
        plt.ylabel('Total Sales Amount ($)')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        return monthly_sales
    
    def create_product_performance_matrix(self):
        """Analyze product performance"""
        merged_data = self.dw.fact_table.merge(
            self.dw.dimension_tables['product_dim'], on='product_id', how='left'
        )
        
        product_performance = merged_data.groupby(['category', 'subcategory', 'product_name']).agg({
            'total_amount': 'sum',
            'quantity': 'sum',
            'sale_id': 'count'
        }).round(2).reset_index()
        
        product_performance.columns = ['Category', 'Subcategory', 'Product', 'Total Revenue', 'Total Quantity', 'Transaction Count']
        
        # Create heatmap
        pivot_table = product_performance.pivot_table(
            values='Total Revenue', 
            index='Category', 
            columns='Subcategory', 
            aggfunc='sum'
        ).fillna(0)
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_table, annot=True, fmt='.0f', cmap='YlOrRd')
        plt.title('Product Performance Heatmap')
        plt.tight_layout()
        plt.show()
        
        return product_performance
    
    def create_customer_segment_analysis(self):
        """Analyze customer segments"""
        merged_data = self.dw.fact_table.merge(
            self.dw.dimension_tables['customer_dim'], on='customer_id', how='left'
        )
        
        segment_analysis = merged_data.groupby('customer_segment').agg({
            'total_amount': ['sum', 'mean', 'count'],
            'customer_id': 'nunique'
        }).round(2)
        
        segment_analysis.columns = ['Total Revenue', 'Average Transaction', 'Transaction Count', 'Unique Customers']
        
        # Create pie chart
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.pie(segment_analysis['Total Revenue'], labels=segment_analysis.index, autopct='%1.1f%%')
        plt.title('Revenue by Customer Segment')
        
        plt.subplot(1, 2, 2)
        plt.bar(segment_analysis.index, segment_analysis['Average Transaction'])
        plt.title('Average Transaction by Segment')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return segment_analysis
    
    def create_region_analysis(self):
        """Analyze regional performance"""
        merged_data = self.dw.fact_table.merge(
            self.dw.dimension_tables['customer_dim'], on='customer_id', how='left'
        ).merge(
            self.dw.dimension_tables['date_dim'], on='date_id', how='left'
        )
        
        region_analysis = merged_data.groupby(['region', 'quarter']).agg({
            'total_amount': 'sum',
            'quantity': 'sum'
        }).unstack().fillna(0)
        
        # Create stacked bar chart
        region_analysis['total_amount'].plot(kind='bar', stacked=True, figsize=(12, 6))
        plt.title('Quarterly Revenue by Region')
        plt.xlabel('Region')
        plt.ylabel('Total Revenue ($)')
        plt.legend(title='Quarter')
        plt.tight_layout()
        plt.show()
        
        return region_analysis

def main():
    print("📊 OLAP Analysis Dashboard")
    
    # Initialize data warehouse
    dw = DataWarehouse()
    dw.generate_sample_data()
    
    # Initialize analyzer
    analyzer = OLAPAnalyzer(dw)
    
    # Perform analyses
    print("1. Sales Trend Analysis...")
    sales_trend = analyzer.create_sales_trend_analysis()
    
    print("2. Product Performance Analysis...")
    product_perf = analyzer.create_product_performance_matrix()
    
    print("3. Customer Segment Analysis...")
    customer_segments = analyzer.create_customer_segment_analysis()
    
    print("4. Regional Analysis...")
    region_analysis = analyzer.create_region_analysis()
    
    print("✅ OLAP analysis completed!")

if __name__ == "__main__":
    main()