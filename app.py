import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pro import load_data, calculate_rfm, get_total_stats, get_customer_details, calculate_clustering_performance, scale_data

# Step 1: Streamlit Configuration - This must be the first Streamlit command
st.set_page_config(page_title="Customer Segmentation", layout="wide")

# Streamlit app configuration
st.title("Customer Segmentation and Clustering Comparison")

# Step 2: Sidebar for File Upload
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload Customer Dataset (Excel)", type=["xlsx"])

if uploaded_file is not None:
    # Display a message while processing the file
    with st.spinner("Processing data..."):
        # Step 3: Load and process data
        data = load_data(uploaded_file)
    st.success("Data uploaded and processed successfully!")

    # Step 3: Display Stats (Total Money Spent, Total Products Sold, Most Sold Item)
    st.subheader("Dataset Overview")
    stats = get_total_stats(data)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Products Sold", stats['total_products'])
    col2.metric("Total Money Spent", f"${stats['total_spent']:.2f}")
    col3.metric("Most Sold Item Quantity", f"{stats['most_sold_quantity']} units", f"In {stats['most_sold_year']}")
    st.write(f"**Most Sold Product:** {stats['most_sold_item']}")

    # Step 3a: Interactive Pie Chart of Products Sold per Month
    st.subheader("Products Sold per Month")
    monthly_product_sales = data.groupby('YearMonth')['Quantity'].sum().reset_index()
    monthly_product_sales['YearMonth'] = monthly_product_sales['YearMonth'].astype(str)
    fig_pie = px.pie(monthly_product_sales, names='YearMonth', values='Quantity', title="Products Sold Distribution per Month")
    st.plotly_chart(fig_pie)

    # Step 3b: Interactive Line Plot of Products Sold per Month
    st.subheader("Trend of Products Sold Over Time")
    fig_line = px.line(monthly_product_sales, x='YearMonth', y='Quantity', title="Monthly Product Sales Trend")
    fig_line.update_layout(xaxis_title="Year-Month", yaxis_title="Number of Products Sold")
    st.plotly_chart(fig_line)

    # Step 4: Display RFM Segmentation Results and Customer Segment Distribution
    st.subheader("Customer Segments")

    # Calculate RFM data
    rfm_data = calculate_rfm(data)

    # Display the RFM table
    st.dataframe(rfm_data[['CustomerID', 'RFM_Score', 'Customer_segment']])

    # Create and display the customer segment distribution pie chart
    segment_counts = rfm_data['Customer_segment'].value_counts()
    
    fig_segment_pie = px.pie(
        names=segment_counts.index,
        values=segment_counts.values,
        title="Customer Segment Distribution"
    )
    
    # Customize the pie chart
    fig_segment_pie.update_traces(
        textinfo='percent+label',
        textposition='inside',
    )
    
    fig_segment_pie.update_layout(
        title_font_size=24,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",      
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        height=600,
        width=1000,
        margin=dict(t=100, b=40, l=40, r=40)
    )
    
    st.plotly_chart(fig_segment_pie, use_container_width=True)

    # Step 5: Customer Search with Detailed Information
    st.subheader("Search for Customer Segment and Details")
    customer_id = st.text_input("Enter Customer ID to check details and get personalized recommendations:")
    
    if st.button("Search Customer"):
        customer_details = get_customer_details(customer_id, data, rfm_data)
        
        if customer_details:
            st.markdown(f"<h3 style='font-size:24px;'>Customer ID: {customer_id}</h3>", unsafe_allow_html=True)
            
            st.markdown(""" 
            <table style='width:100%; font-size:18px;'>
                <tr><td><strong>Total Purchases:</strong></td><td>{total_purchases}</td></tr>
                <tr><td><strong>Total Amount Spent:</strong></td><td>${total_spent:.2f}</td></tr>
                <tr><td><strong>Customer Segment:</strong></td><td>{segment}</td></tr>
            </table>
            """.format(
                total_purchases=customer_details['total_purchases'],
                total_spent=customer_details['total_spent'],
                segment=customer_details['segment']
            ), unsafe_allow_html=True)

            st.markdown("<h4>Items Purchased:</h4>", unsafe_allow_html=True)
            st.markdown("<div style='max-height: 200px; overflow-y: scroll; padding: 10px; border: 1px solid #ddd; border-radius: 4px;'>", unsafe_allow_html=True)
            for item in customer_details['items_purchased']:
                st.markdown(f"<p style='margin: 5px 0;'>{item}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Purchase pattern graph
            st.subheader("Purchase Pattern")
            purchase_pattern = customer_details['purchase_pattern']
            fig_purchase_pattern = go.Figure()
            fig_purchase_pattern.add_trace(go.Bar(
                x=purchase_pattern['Date'], y=purchase_pattern['Quantity'], name='Quantity Purchased'
            ))
            fig_purchase_pattern.update_layout(
                title="Customer's Purchase Pattern Over Time",
                xaxis_title="Date",
                yaxis_title="Quantity",
                font=dict(size=16)
            )
            st.plotly_chart(fig_purchase_pattern)

            # Enhanced recommendation display in pointwise format
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

            # Display recommendation based on segment
            segment = customer_details['segment']
            if segment in recommendations:
                rec = recommendations[segment]
                st.markdown(f"<h4 style='font-size:22px; color:#ff5733;'>Personalized Recommendations for {segment}</h4>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:18px;'><strong>Improvement Focus:</strong> {rec['improvement']}</p>", unsafe_allow_html=True)
                st.markdown("<ul>", unsafe_allow_html=True)
                for offer in rec['offers']:
                    st.markdown(f"<li style='font-size:18px;'>{offer}</li>", unsafe_allow_html=True)
                st.markdown("</ul>", unsafe_allow_html=True)

                st.markdown(f"<p style='font-size:18px;'><strong>{rec['personal_message']}</strong></p>", unsafe_allow_html=True)

    # Step 6: Clustering Comparison
    if st.button("Run Clustering Comparison"):
        X_scaled = scale_data(rfm_data[['Recency', 'Frequency', 'Monetary']])
        clustering_results, best_labels, cluster_profiles = calculate_clustering_performance(X_scaled, rfm_data)
        
        rfm_data['Customer Type'] = [cluster_profiles.loc[cluster, 'Customer Type'] for cluster in best_labels]

        best_algorithm = clustering_results.iloc[clustering_results['Silhouette Score'].idxmax()]['Algorithm']
        st.markdown(f"<h3 style='font-size:30px; color:#ff5733;'>Best Algorithm: {best_algorithm}</h3>", unsafe_allow_html=True)

        
        fig_3d = px.scatter_3d(
            rfm_data,
            x='Recency',
            y='Frequency',
            z='Monetary',
            color='Customer Type',
            title=f"3D Clustering Result with Customer Types",
            labels={'x': 'Recency', 'y': 'Frequency', 'z': 'Monetary'}
        )
        fig_3d.update_layout(width=900, height=700)
        st.plotly_chart(fig_3d)
        
        st.subheader("Clustering Algorithm Comparison Results")
        st.dataframe(clustering_results[['Algorithm', 'Silhouette Score', 'Time Taken (seconds)', 'Number of Clusters', 'Notes']])
        
        st.subheader("Cluster Profiles with Customer Types")
        st.dataframe(cluster_profiles)

else:
    st.warning("Please upload a dataset to proceed with customer segmentation analysis.")
