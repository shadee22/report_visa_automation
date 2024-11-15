import streamlit as st
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker


def load_datasets():
    # Sidebar file uploaders
    st.sidebar.write("### Upload CSV Files (Required)")
    card_file = st.sidebar.file_uploader("Upload Visa Japan Cards CSV", type="csv")
    transaction_file = st.sidebar.file_uploader("Upload Visa Japan Eligible Transactions CSV", type="csv")
    redemption_file = st.sidebar.file_uploader("Upload Visa Japan Redemptions CSV", type="csv")

    # Check if all required files are uploaded
    if card_file is not None and transaction_file is not None and redemption_file is not None:
        card_df = pd.read_csv(card_file)
        transaction_df = pd.read_csv(transaction_file)
        redemption_df = pd.read_csv(redemption_file)
        return card_df, transaction_df, redemption_df
    else:
        return None, None, None

def html_spacer(lines=1):
    st.markdown("<br>" * lines, unsafe_allow_html=True)

def display_card_count_analysis(cards_df):
    generated_time = datetime.now().strftime('%H:%M:%S JST')
    st.write(f"Generated @ {generated_time}")
    df=  cards_df
    num_unique_cardholders = df['cardholder_id'].nunique()
    num_unique_cards = df['card_id'].nunique()
    st.metric("Unique Cardholders", f"{num_unique_cardholders:,}")
    st.metric("Unique Cards", f"{num_unique_cards:,}")
    card_counts = df.groupby('cardholder_id')['card_id'].count().reset_index(name='card_id_count')
    value_counts = card_counts['card_id_count'].value_counts().reset_index()
    value_counts.columns = ['card_count', 'unique_cardholder_count']
    value_counts['percentage_of_total'] = ((value_counts['unique_cardholder_count'] / num_unique_cardholders) * 100).round(2).astype(str) + '%'
    st.write("### Card Count Analysis Table")
    st.table(value_counts.rename(columns={
        'card_count': 'Card Count',
        'unique_cardholder_count': 'Unique Cardholder Count',
        'percentage_of_total': 'Percentage'
    }))

def display_daily_cardholder_enrollment_plot(card_df, filename='daily_enrollment_plot.png'):
    html_spacer(8)


    st.write("### Daily Cardholder Enrollment Plot")
# Define two columns
    col1, col2 = st.columns(2)

    # Display the last and first card enrollment dates
    with col1:
        st.success(f"Last Card Enrollment Date: {card_df['created_at'].max()}")

    with col2:
        st.success(f"First Card Enrollment Date: {card_df['created_at'].min()}")

    # st.write("card-df max and min date", card_df['created_at'].max(), card_df['created_at'].min())

    # Ensure 'created_at' column exists
    if 'created_at' not in card_df.columns:
        st.error("'created_at' column is missing in the dataset.")
        return

    # Ensure 'created_at' is in datetime format
    card_df['created_at'] = pd.to_datetime(card_df['created_at'], errors='coerce')

    # Filter rows with valid dates
    card_df = card_df.dropna(subset=['created_at'])

    # Convert 'created_at' to just the date part
    card_df['date'] = card_df['created_at'].dt.date

    # Get the date 7 days ago from today
    today = datetime.now().date()
    last_7_days = today - timedelta(days=7)

    # Filter the DataFrame for the last 7 days excluding today
    df_last_7_days = card_df[(card_df['date'] >= last_7_days) & (card_df['date'] < today)]

    # Handle empty data scenario
    if df_last_7_days.empty:
        st.warning("No data available for the last 7 days to plot daily enrollment.")
        return

    # Group by date and count the number of unique cardholder_id per day
    daily_enrollment = df_last_7_days.groupby('date')['cardholder_id'].nunique()

    # Handle empty grouping scenario
    if daily_enrollment.empty:
        st.warning("No enrollment data found for the last 7 days.")
        return

    # Plotting
    plt.figure(figsize=(10, 6))
    ax = daily_enrollment.plot(kind='bar', color='skyblue')  # Adjust color as needed
    plt.xlabel('Date', fontsize=8)
    plt.ylabel('Number of Unique Cardholders', fontsize=8)
    plt.title('Daily Cardholder Enrollment for the Last 7 Days', fontsize=10)
    plt.xticks(rotation=45, fontsize=8)
    plt.yticks(fontsize=8)

    # Add value labels on top of each bar with smaller font size
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=7, xytext=(0, 5), textcoords='offset points')

    plt.tight_layout()

    # Save the plot
    plt.savefig(filename)
    plt.close()

    # Display the plot in Streamlit
    st.image(filename)


def display_cardholder_distribution_by_card_count(df):
    html_spacer(20)

    st.write("### Cardholder Distribution by Card Count")

    # Group by cardholder and count the number of cards per cardholder
    card_counts = df.groupby('cardholder_id')['card_id'].count().reset_index(name='card_id_count')

    # Count occurrences of each card count value
    value_counts = card_counts['card_id_count'].value_counts().reset_index()
    value_counts.columns = ['card_count', 'unique_cardholder_count']

    # Define a color palette
    blue_palette = plt.cm.Blues(np.linspace(0.9, 0.3, len(value_counts)))

    # Plotting the donut chart
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        value_counts['unique_cardholder_count'], 
        labels=value_counts['card_count'], 
        autopct='%1.1f%%', 
        textprops=dict(color="w"), 
        colors=blue_palette,
        wedgeprops=dict(width=0.4)  # Adjust the width for a donut shape
    )

    # Add legend
    ax.legend(wedges, [f"{count} Card(s): {holders:,} Cardholders" 
                       for count, holders in zip(value_counts['card_count'], value_counts['unique_cardholder_count'])],
              title="Card Distribution",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))

    # Add a title
    plt.title('Cardholder Distribution by Card Count', fontsize=16)

    # Display the plot in Streamlit
    st.pyplot(fig)

def display_top_issuers_analysis(df):
    html_spacer(21)

    st.write("### Top Issuers Analysis (Top 22)")

    # Convert 'issuer_bin' to string and extract the first 4 digits
    df['issuer_bin'] = df['issuer_bin'].astype(str)
    df['first_4_digits'] = df['issuer_bin'].str[:4]

    # Group by the first 4 digits and sum the card counts per issuer
    issuer_counts = df.groupby('first_4_digits')['card_id'].nunique().reset_index(name='card_id_count')
    
    # Sort issuers by card count in descending order and select the top 22
    issuer_counts = issuer_counts.sort_values(by='card_id_count', ascending=False).head(22)

    # Bank mapping dictionary
    bank_mapping = {
        '4980': 'SUMITOMO MITSUI CARD COMPANY LIMITED',
        '4708': 'YES BANK, LTD.',
        '4297': 'RAKUTEN KC CO., LTD.',
        '4537': 'WELLS FARGO BANK, N.A.',
        '4205': 'AEON CREDIT SERVICE CO., LTD.',
        '4534': 'DC CARD CO., LTD.',
        '4541': 'CREDIT SAISON CO., LTD.',
        '4363': 'UNITED COMMERCIAL BANK',
        '4649': 'YAMAGIN CREDIT CO., LTD.-4649',
        '4616': 'U.S. BANK N.A. ND',
        '4986': 'MITSUBISHI UFJ FINANCIAL GROUP, INC.-4986',
        '4539': 'OSTGIROT BANK AB',
        '4097': 'CAJA AHORROS GERONA',
        '4924': 'BANK OF AMERICA, N.A.',
        '4987': 'YAMAGIN CREDIT CO., LTD.-4987',
        '4162': 'UNKNOWN',
        '4721': 'WELLS FARGO BANK IOWA, N.A.',
        '4538': 'MITSUBISHI UFJ FINANCIAL GROUP, INC.-4538',
        '4984': 'BANCO DO BRASIL, S.A.',
        '4122': 'UNITED BANK, LTD.',
        '4901': 'OMC CARD, INC.',
        '4624': 'BROADWAY BANK'
    }

    # Map 'first_4_digits' to 'Bank Name' using the mapping dictionary
    issuer_counts['Bank Name'] = issuer_counts['first_4_digits'].map(bank_mapping).fillna('<unknown>')

    # Reshape the DataFrame columns for display
    issuer_counts = issuer_counts[['Bank Name', 'card_id_count']].rename(columns={'card_id_count': 'Card ID Count'})

    # Display the table in Streamlit
    st.table(issuer_counts)


def display_top_issuers_analysis_plot(df):
    html_spacer(5)
    st.write("### Top Issuers Analysis")

    # Convert 'issuer_bin' to string and extract the first 4 digits
    df['issuer_bin'] = df['issuer_bin'].astype(str)
    df['first_4_digits'] = df['issuer_bin'].str[:4]

    # Group by the first 4 digits and sum the card counts per issuer
    issuer_counts = df.groupby('first_4_digits')['card_id'].nunique().reset_index(name='card_id_count')
    
    # Sort issuers by card count in descending order and select the top 22
    issuer_counts = issuer_counts.sort_values(by='card_id_count', ascending=False).head(22)

    # Bank mapping dictionary
    bank_mapping = {
        '4980': 'SUMITOMO MITSUI CARD COMPANY LIMITED',
        '4708': 'YES BANK, LTD.',
        '4297': 'RAKUTEN KC CO., LTD.',
        '4537': 'WELLS FARGO BANK, N.A.',
        '4205': 'AEON CREDIT SERVICE CO., LTD.',
        '4534': 'DC CARD CO., LTD.',
        '4541': 'CREDIT SAISON CO., LTD.',
        '4363': 'UNITED COMMERCIAL BANK',
        '4649': 'YAMAGIN CREDIT CO., LTD.-4649',
        '4616': 'U.S. BANK N.A. ND',
        '4986': 'MITSUBISHI UFJ FINANCIAL GROUP, INC.-4986',
        '4539': 'OSTGIROT BANK AB',
        '4097': 'CAJA AHORROS GERONA',
        '4924': 'BANK OF AMERICA, N.A.',
        '4987': 'YAMAGIN CREDIT CO., LTD.-4987',
        '4162': 'UNKNOWN',
        '4721': 'WELLS FARGO BANK IOWA, N.A.',
        '4538': 'MITSUBISHI UFJ FINANCIAL GROUP, INC.-4538',
        '4984': 'BANCO DO BRASIL, S.A.',
        '4122': 'UNITED BANK, LTD.',
        '4901': 'OMC CARD, INC.',
        '4624': 'BROADWAY BANK'
    }

    # Map 'first_4_digits' to Bank Name
    issuer_counts['Bank Name'] = issuer_counts['first_4_digits'].map(bank_mapping).fillna('<unknown>')

    # Sort by card count and prepare for plotting
    issuer_counts = issuer_counts[['Bank Name', 'card_id_count']].rename(columns={'card_id_count': 'Card ID Count'})
    issuer_counts = issuer_counts.sort_values(by='Card ID Count', ascending=True)  # For horizontal bar plot

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(issuer_counts['Bank Name'], issuer_counts['Card ID Count'], color='skyblue')
    ax.set_xlabel('Card ID Count')
    ax.set_ylabel('Bank Name')
    ax.set_title('Top Issuers by Card ID Count')

    # Display the plot in Streamlit
    st.pyplot(fig)
    
    
def display_authorized_eligible_transactions_analysis(df):
    # Current timestamp for display
    generated_time = datetime.now().strftime('%H:%M:%S JST')

    st.write(f"### Authorized & Eligible Transactions Analysis (Generated @ {generated_time})")

    # Calculate metrics
    total_transactions = df['transaction_id'].nunique()
    total_transaction_value = df['transaction_amount'].sum()
    avg_transaction_value = df['transaction_amount'].mean()
    total_cashback = df['cashback_amount'].sum()
    avg_cashback = df['cashback_amount'].mean()

    # Create a summary table
    metrics_data = {
        'Metrics': [
            'Authorized & Eligible Total Transaction Count',
            'Authorized & Eligible Total Transaction Value',
            'Authorized & Eligible Average Transaction Value',
            'Authorized & Eligible Total Cashback',
            'Authorized & Eligible Average Cashback'
        ],
        'Value': [
            f"{total_transactions:,}",  # Add comma formatting
            f"¥{total_transaction_value:,.2f}",  # Yen symbol and 2 decimal places
            f"¥{avg_transaction_value:,.2f}",
            f"¥{total_cashback:,.2f}",
            f"¥{avg_cashback:,.2f}"
        ]
    }

    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics_data)

    # Display the table in Streamlit
    st.table(metrics_df)


def display_redemption_analysis(df):
    # Current timestamp for display
    generated_time = datetime.now().strftime('%H:%M:%S JST')

    st.write(f"### Redemption Analysis (Generated @ {generated_time})")

    # Calculate metrics
    total_redemptions = df['transaction_id'].nunique()
    total_cashback_given = df['cashback_amount'].sum()
    avg_cashback_given = df['cashback_amount'].mean()

    # Create a summary table
    metrics_data = {
        'Metrics': [
            'Total Redemption Count',
            'Total Cashback Given',
            'Average Cashback Given'
        ],
        'Value': [
            f"{total_redemptions:,}",  # Add comma formatting
            f"¥{total_cashback_given:,.2f}",  # Yen symbol and 2 decimal places
            f"¥{avg_cashback_given:,.2f}"
        ]
    }

    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics_data)

    # Display the table in Streamlit
    st.table(metrics_df)

def display_daily_redemptions_value_plot(redemption_df):
    html_spacer(3)
    st.write("### Daily Redemptions Value Plot")

    # Ensure 'created_at' and 'cashback_amount' columns exist
    if 'created_at' not in redemption_df.columns or 'cashback_amount' not in redemption_df.columns:
        st.error("Required columns ('created_at' and 'cashback_amount') are missing in the dataset.")
        return

    # Convert 'created_at' to datetime format
    redemption_df['created_at'] = pd.to_datetime(redemption_df['created_at'], errors='coerce')

    # Filter rows with valid dates
    redemption_df = redemption_df.dropna(subset=['created_at'])

    # Convert 'created_at' to just the date part
    redemption_df['date'] = redemption_df['created_at'].dt.date

    # Get the date 7 days ago from today
    today = datetime.now().date()
    last_7_days = today - timedelta(days=7)

    # Filter the DataFrame for the last 7 days excluding today
    df_last_7_days = redemption_df[(redemption_df['date'] >= last_7_days) & (redemption_df['date'] < today)]

    # Handle empty data scenario
    if df_last_7_days.empty:
        st.warning("No redemption data available for the last 7 days.")
        return

    # Group by date and sum the redemption value per day
    daily_redemptions_value = df_last_7_days.groupby('date')['cashback_amount'].sum()

    # Plotting
    plt.figure(figsize=(10, 6))
    ax = daily_redemptions_value.plot(kind='bar', color='blue')  # Adjust color as needed
    plt.xlabel('Date', fontsize=8)
    plt.ylabel('Redemption Value', fontsize=8)
    plt.title('Daily Redemptions Value for the Last 7 Days', fontsize=10)
    plt.xticks(rotation=45, fontsize=8)
    plt.yticks(fontsize=8)

    # Add value labels on top of each bar with smaller font size
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=7, xytext=(0, 5), textcoords='offset points')

    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(plt)


def display_daily_redemptions_count_plot(redemption_df):
    st.write("### Daily Redemptions Count Plot")

    # Ensure 'created_at' column exists
    if 'created_at' not in redemption_df.columns:
        st.error("'created_at' column is missing in the dataset.")
        return

    # Convert 'created_at' to datetime format
    redemption_df['created_at'] = pd.to_datetime(redemption_df['created_at'], errors='coerce')

    # Filter rows with valid dates
    redemption_df = redemption_df.dropna(subset=['created_at'])

    # Convert 'created_at' to just the date part
    redemption_df['date'] = redemption_df['created_at'].dt.date

    # Get the date 7 days ago from today
    today = datetime.now().date()
    last_7_days = today - timedelta(days=7)

    # Filter the DataFrame for the last 7 days excluding today
    df_last_7_days = redemption_df[(redemption_df['date'] >= last_7_days) & (redemption_df['date'] < today)]

    # Handle empty data scenario
    if df_last_7_days.empty:
        st.warning("No redemption data available for the last 7 days.")
        return

    # Group by date and count the number of redemptions per day
    daily_redemptions_count = df_last_7_days.groupby('date').size()

    # Plotting
    plt.figure(figsize=(10, 6))
    ax = daily_redemptions_count.plot(kind='bar', color='darkblue')  # Adjust color as needed
    plt.xlabel('Date', fontsize=8)
    plt.ylabel('Number of Redemptions', fontsize=8)
    plt.title('Daily Redemptions Count for the Last 7 Days', fontsize=10)
    plt.xticks(rotation=45, fontsize=8)
    plt.yticks(fontsize=8)

    # Add value labels on top of each bar with smaller font size
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=7, xytext=(0, 5), textcoords='offset points')

    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(plt)

def display_merchant_wise_redemption_analysis(redemption_df):
    # Current timestamp for display
    generated_time = datetime.now().strftime('%H:%M:%S JST')
    html_spacer(3)
    st.write(f"### Merchant-wise Total Redemption Analysis (Generated @ {generated_time})")
    # Ensure 'name' and 'cashback_amount' columns exist
    if 'name' not in redemption_df.columns or 'cashback_amount' not in redemption_df.columns:
        st.error("Required columns ('name' and 'cashback_amount') are missing in the dataset.")
        return

    # Group by merchant and sum the redemption value per merchant
    merchant_redemption = redemption_df.groupby('name')['cashback_amount'].sum().reset_index()

    # Sort merchants by redemption value in descending order
    merchant_redemption = merchant_redemption.sort_values(by='cashback_amount', ascending=False)

    # Format the 'cashback_amount' as currency for display
    merchant_redemption['cashback_amount'] = merchant_redemption['cashback_amount'].apply(lambda x: f"¥{x:,.0f}")

    # Rename columns for display
    merchant_redemption = merchant_redemption.rename(columns={'name': 'Merchant', 'cashback_amount': 'Value'})

    # Display the table in Streamlit
    st.table(merchant_redemption.reset_index(drop=True))


def display_total_redemptions_value_by_merchants_plot(redemption_df):
    html_spacer(18)
    st.write("### Total Redemptions Value By Merchants")

    # Ensure 'name' and 'cashback_amount' columns exist
    if 'name' not in redemption_df.columns or 'cashback_amount' not in redemption_df.columns:
        st.error("Required columns ('name' and 'cashback_amount') are missing in the dataset.")
        return

    # Group by merchant and sum the redemption value per merchant
    merchant_redemption = redemption_df.groupby('name')['cashback_amount'].sum().reset_index()

    # Sort merchants by redemption value in descending order
    merchant_redemption = merchant_redemption.sort_values(by='cashback_amount', ascending=False)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(merchant_redemption['name'], merchant_redemption['cashback_amount'], color=plt.cm.Blues(np.linspace(0.9, 0.3, len(merchant_redemption))))
    ax.set_xlabel('Merchant Name')
    ax.set_ylabel('Sum of Cashback')
    ax.set_title('Total Redemption By Merchants')
    plt.xticks(rotation=45, ha='right')

    # Format the y-axis to display in millions with two decimal places
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x / 1e6:.2f}M'))

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height / 1e6:.2f}M',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),  # Offset text above bar
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    # Display the plot in Streamlit
    st.pyplot(fig)


def main():
    today_date = datetime.now().strftime('%B %d, %Y')
    st.title(f"Visa Japan Campaign Analysis {today_date}")

    # Load datasets from either uploaded files or default paths
    card_df, transaction_df, redemption_df = load_datasets()

    # Check if all required files are uploaded
    if card_df is not None and transaction_df is not None and redemption_df is not None:
        display_card_count_analysis(card_df)
        display_daily_cardholder_enrollment_plot(card_df)
        display_cardholder_distribution_by_card_count(card_df)
        display_top_issuers_analysis(card_df)
        display_top_issuers_analysis_plot(card_df)
        display_authorized_eligible_transactions_analysis(transaction_df)
        display_redemption_analysis(redemption_df)
        display_daily_redemptions_value_plot(redemption_df)
        display_daily_redemptions_count_plot(redemption_df)
        display_merchant_wise_redemption_analysis(redemption_df)
        display_total_redemptions_value_by_merchants_plot(redemption_df)
    else:
        st.warning("Please upload all required CSV files in the sidebar to generate insights.")


if __name__ == "__main__":
    main()
