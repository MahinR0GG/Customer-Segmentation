import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import time

sns.set()

def load_data(file_path):
    try:
        data = pd.read_excel(file_path)
        data['CustomerID'].replace('', np.nan, inplace=True)
        data.dropna(subset=['CustomerID'], inplace=True)
        data.drop_duplicates(inplace=True)  # Fix to remove duplicates
        data['TotalPrice'] = data['Quantity'] * data['UnitPrice']
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def get_total_stats(data):
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    data['Year'] = data['InvoiceDate'].dt.year
    data['YearMonth'] = data['InvoiceDate'].dt.to_period('M')
    
    products_sold_per_month = data.groupby('YearMonth')['Quantity'].sum().reset_index()
    total_money_spent = data.groupby('YearMonth')['TotalPrice'].sum().reset_index()

    most_sold_item_data = data.groupby(['StockCode', 'Description', 'Year'])['Quantity'].sum().reset_index()
    most_sold_item = most_sold_item_data.loc[most_sold_item_data['Quantity'].idxmax()]

    stats = {
        'total_products': products_sold_per_month['Quantity'].sum(),
        'total_spent': total_money_spent['TotalPrice'].sum(),
        'most_sold_item': most_sold_item['Description'],
        'most_sold_quantity': most_sold_item['Quantity'],
        'most_sold_year': most_sold_item['Year']
    }
    return stats

def calculate_rfm(data):
    data_recency = data.groupby(by='CustomerID', as_index=False)['InvoiceDate'].max()
    data_recency.columns = ['CustomerID', 'LastInvoiceDate']
    recent_date = data_recency['LastInvoiceDate'].max()
    data_recency['Recency'] = data_recency['LastInvoiceDate'].apply(lambda x: (recent_date - x).days)

    frequency_data = data.groupby(by=['CustomerID'], as_index=False)['InvoiceDate'].count()
    frequency_data.columns = ['CustomerID', 'Frequency']

    monetary_data = data.groupby(by='CustomerID', as_index=False)['TotalPrice'].sum()
    monetary_data.columns = ['CustomerID', 'Monetary']

    rf_data = data_recency.merge(frequency_data, on='CustomerID')
    rfm_data = rf_data.merge(monetary_data, on='CustomerID').drop(columns='LastInvoiceDate')

    rfm_data['R_rank'] = rfm_data['Recency'].rank(ascending=False)
    rfm_data['F_rank'] = rfm_data['Frequency'].rank(ascending=True)
    rfm_data['M_rank'] = rfm_data['Monetary'].rank(ascending=True)

    rfm_data['R_rank_norm'] = (rfm_data['R_rank'] / rfm_data['R_rank'].max()) * 100
    rfm_data['F_rank_norm'] = (rfm_data['F_rank'] / rfm_data['F_rank'].max()) * 100
    rfm_data['M_rank_norm'] = (rfm_data['F_rank'] / rfm_data['M_rank'].max()) * 100

    rfm_data.drop(columns=['R_rank', 'F_rank', 'M_rank'], inplace=True)
    rfm_data['RFM_Score'] = 0.15 * rfm_data['R_rank_norm'] + 0.28 * rfm_data['F_rank_norm'] + 0.57 * rfm_data['M_rank_norm']
    rfm_data['RFM_Score'] *= 0.05
    rfm_data = rfm_data.round(2)

    rfm_data["Customer_segment"] = np.where(rfm_data['RFM_Score'] > 4.5, "Top Customers",
                                             np.where(rfm_data['RFM_Score'] > 4, "High Value Customer",
                                                      np.where(rfm_data['RFM_Score'] > 3, "Medium Value Customer",
                                                               np.where(rfm_data['RFM_Score'] > 1.6, 'Low Value Customers', 'Lost Customers'))))
    
    return rfm_data

def plot_pie_chart(rfm_data):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(rfm_data['Customer_segment'].value_counts(), labels=rfm_data['Customer_segment'].value_counts().index, autopct='%1.1f%%', startangle=140)
    ax.set_title('Customer Segments Distribution')
    return fig

def search_customer_by_id(customer_id, rfm_data):
    customer_row = rfm_data[rfm_data['CustomerID'] == customer_id]
    if not customer_row.empty:
        customer_segment = customer_row['Customer_segment'].values[0]
        return f"Customer ID {customer_id} belongs to the '{customer_segment}' segment."
    else:
        return f"Customer ID {customer_id} not found."

def get_customer_details(customer_id, data, rfm_data):
    try:
        customer_id = int(customer_id)
        if customer_id in data['CustomerID'].values:
            customer_data = data[data['CustomerID'] == customer_id]
            total_purchases = customer_data['InvoiceNo'].nunique()
            total_spent = customer_data['TotalPrice'].sum()
            items_purchased = customer_data['Description'].unique().tolist()
            customer_segment = rfm_data[rfm_data['CustomerID'] == customer_id]['Customer_segment'].values[0]
            purchase_pattern = customer_data.groupby('InvoiceDate')['Quantity'].sum().reset_index()
            purchase_pattern.columns = ['Date', 'Quantity']

            recommendation = generate_recommendation(customer_segment)
            return {
                'total_purchases': total_purchases,
                'total_spent': total_spent,
                'items_purchased': items_purchased,
                'segment': customer_segment,
                'purchase_pattern': purchase_pattern,
                'recommendation': recommendation
            }
        else:
            return None
    except ValueError:
        return None

def generate_recommendation(segment):
    recommendations = {
        "Top Customers": {
            'improvement': "creating a VIP experience tailored just for them",
            'offers': [
                "Premium discounts, loyalty points, and early access to sales",
                "Invitation to VIP-only events and exclusive previews of new collections",
                "Complimentary shipping or premium delivery options for a seamless experience",
                "Personalized thank-you notes or gifts to celebrate special milestones (e.g., anniversaries, birthdays)"
            ],
            'personal_message': "We value your loyalty and commitment. Here are some exclusive rewards to make your shopping experience even more delightful!"
        },
        "High Value Customer": {
            'improvement': "offering a curated and personalized shopping experience",
            'offers': [
                "Special discounts, member-only benefits, and tailored product recommendations",
                "Access to a dedicated personal shopper or virtual stylist service",
                "Early notifications on items that match their preferences",
                "Priority access to limited-edition products or collections"
            ],
            'personal_message': "Thank you for being a valued customer! We've curated some special offers just for you to make every shopping experience memorable."
        },
        "Medium Value Customer": {
            'improvement': "boosting engagement with enticing and time-sensitive offers",
            'offers': [
                "Discount coupons, seasonal offers, and new product previews",
                "Limited-time bundles and value packs for popular items",
                "Loyalty rewards that can be accumulated and redeemed over time",
                "Encouraging frequent purchases with a points-based reward system"
            ],
            'personal_message': "We're glad to have you with us! Here are some great deals and offers to help you get the most value from your shopping."
        },
        "Low Value Customers": {
            'improvement': "attracting interest with introductory and first-time discounts",
            'offers': [
                "First-time discounts, loyalty rewards, and introductory offers",
                "Special promotions on best-selling items to encourage repeat purchases",
                "Personalized recommendations on affordable but popular products",
                "Flexible return policies or satisfaction guarantees to build trust"
            ],
            'personal_message': "Welcome! As a thank you for shopping with us, we’re excited to offer you some special deals to help you discover what we have to offer."
        },
        "Lost Customers": {
            'improvement': "re-engaging them with valuable promotions and loyalty incentives",
            'offers': [
                "General promotions and event-based discounts",
                "Special 'We Miss You' discounts or one-time comeback offers",
                "Personalized email reminders about items they may have liked in the past",
                "Loyalty program enrollment bonuses to encourage re-engagement"
            ],
            'personal_message': "We miss you! Here are some exclusive offers and deals to welcome you back. Rediscover the things you love with us!"
        }
    }
    # Return recommendations for the specified segment
    return recommendations.get(segment, {"improvement": "None", "offers": ["No offers available."], "personal_message": "No message available."})

def scale_data(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)

def calculate_clustering_performance(X_scaled, rfm_data):
    results = {
        'Algorithm': [],
        'Silhouette Score': [],
        'Time Taken (seconds)': [],
        'Number of Clusters': [],
        'Notes': []
    }
    
    best_labels = None
    best_score = -1
    best_algorithm = None
    
    clustering_algorithms = {
        "KMeans": KMeans(n_clusters=3, random_state=42),
        "DBSCAN": DBSCAN(eps=0.3, min_samples=10),
        "Gaussian Mixture": GaussianMixture(n_components=3, random_state=42)
    }
    
    for name, model in clustering_algorithms.items():
        start_time = time.time()
        
        # Fit model and predict labels
        if name == "Gaussian Mixture":
            labels = model.fit_predict(X_scaled)
        else:
            labels = model.fit(X_scaled).labels_
        
        # Calculate silhouette score
        unique_labels = set(labels)
        if name == "DBSCAN":
            # For DBSCAN, exclude noise points (-1) from the silhouette score calculation
            valid_labels = [label for label in unique_labels if label != -1]
            if len(valid_labels) > 1:
                score = silhouette_score(X_scaled[labels != -1], labels[labels != -1])
                note = ""
            else:
                score = np.nan
                note = "High noise rate"
        else:
            # For other algorithms, use all the labels
            score = silhouette_score(X_scaled, labels)
            note = ""
        
        elapsed_time = time.time() - start_time
        cluster_count = len(unique_labels)
        
        if score > best_score:
            best_score = score
            best_labels = labels
            best_algorithm = name
        
        results['Algorithm'].append(name)
        results['Silhouette Score'].append(score)
        results['Time Taken (seconds)'].append(elapsed_time)
        results['Number of Clusters'].append(cluster_count)
        results['Notes'].append(note)
    
    # Calculate cluster profiles for the best labels
    rfm_data['Cluster'] = best_labels
    numeric_rfm_data = rfm_data.select_dtypes(include=[np.number])  # Only use numeric columns
    cluster_profiles = numeric_rfm_data.groupby(rfm_data['Cluster']).mean()
    cluster_profiles['Customer Type'] = cluster_profiles.apply(
        lambda row: "Top Value" if row['Monetary'] > rfm_data['Monetary'].quantile(0.75) else
                    "High Value" if row['Monetary'] > rfm_data['Monetary'].quantile(0.5) else
                    "Medium Value" if row['Monetary'] > rfm_data['Monetary'].quantile(0.25) else
                    "Low Value", axis=1
    )
    
    return pd.DataFrame(results), best_labels, cluster_profiles
