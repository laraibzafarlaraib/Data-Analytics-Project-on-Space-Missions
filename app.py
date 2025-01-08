import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud


# Set the page title and icon
st.set_page_config(
    page_title="Space Missions Explorer", 
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
    )

# Sidebar navigation
st.sidebar.title("Explore Analysis")

options = ["Home",
    "Gallery",
    "Analysis among differernt features", 
    "Feature distribution analysis",
    "Missing Values Analysis",
    "Outliers Detection",
    "Machine Learning"
    ]

selection = st.sidebar.radio("🚀Choose an analysis:", options)

# Load dataset
df = pd.read_csv('space_missions.csv')  # Replace with your dataset file

#home
if selection == "Home":
    st.title("Space Missions Analysis")

    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            Welcome to the Space Missions Explorer! 🚀 Explore humanity's journey to the stars. 
            Here you will be able to explore Mankind at its peak, with history of missions in space and outstanding achievments. 

            Take a look at mission details starting from 1953, view breathtaking images, and select different kind of analysis on space exploration datatset. 

            Start your interstellar journey today!
            """
        )

    with col2:
        st.image(
            "https://live.staticflickr.com/65535/53612339128_672b5bc6ab.jpg", 
            caption="The Final Frontier",
            use_container_width=True
        )
    st.subheader("Dataset Overview")
    st.dataframe(df)

elif selection == "Missing Values Analysis":
    st.title("Analysis Details")
# Analysis among differernt features page
    missing_values = df.isnull().sum()

    missing_data = {'Columns': ['Company', 'Location', 'Year', 'Time', 'Rocket', 'MissionStatus', 'RocketStatus', 'Price', 'Mission'],
                    'Missing Count': [0, 0, 0, 125, 0, 0, 0, 3362, 0]}
    missing_df = pd.DataFrame(missing_data)

    # Bar plot with plotly
    fig = px.bar(missing_df, x='Columns', y='Missing Count', title='Missing Values Per Column',
                text='Missing Count', color='Missing Count', color_continuous_scale='viridis')
    fig.update_traces(textposition='outside')
    fig.update_layout(xaxis_title='Columns', yaxis_title='Missing Count', coloraxis_showscale=False)
    st.plotly_chart(fig)

elif selection == "Outliers Detection":
    st.title("Analysis Details")

        # Create a boxplot for the 'Price' column to detect outliers
    fig = px.box(df, y="Price", title="Outlier Detection for Price", 
                labels={"Price": "Price"}, color_discrete_sequence=['#FF6347'])

    st.plotly_chart(fig)




elif selection == "Analysis among differernt features":
    st.title("Analysis Details")
    mission_list = [
        "Global Space Mission Trends Over Time",
        "Distribution of Rocket Prices", 
        "Mission Success Rate by Company", 
        "Top 10 Companies by Mission Count",
        "Number of Missions by Country",
        "Price Distribution by Rocket Status", 
        "Mission Trends and Rocket Prices Over Years",
        "Correlation analysis"
        ]
    
    selected_mission = st.selectbox("Select an Analysis:", mission_list)

    mission_info = {
        "Global Space Mission Trends Over Time": "Visualize the number of space missions conducted annually over time.",
        "Distribution of Rocket Prices": "Analyze the variation in rocket launch costs across different missions.",
        "Mission Success Rate by Company": "Examine the success rate of space missions conducted by various companies.",
        "Top 10 Companies by Mission Count": " Identify the leading companies based on the number of missions they have conducted.",
        "Number of Missions by Country": "Explore the distribution of space missions conducted by different countries.",
        "Price Distribution by Rocket Status": "Compare the cost of rockets based on their operational status (active, retired, etc.).",
        "Mission Trends and Rocket Prices Over Years": "Study the correlation between space mission trends and changes in rocket prices over time.",
        "Correlation analysis" : "Year and MissionStatus (0.14): This represents a weak positive correlation. It suggests that as the year increases, there is a slight improvement in mission status. The relationship is not strong and could be influenced by other factors.MissionStatus and Price (0.05): This is a very weak positive correlation, close to zero. It indicates almost no linear relationship between mission status and price. A higher mission status does not appear to be strongly associated with changes in price.Year and Price (-0.41): This shows a moderate negative correlation. It suggests that as the years progress, the prices tend to decrease. This could imply advancements in technology, better cost management, or other industry trends leading to lower costs over time.The diagonal always shows 1.0 since each variable is perfectly correlated with itself."
    }

    st.subheader(selected_mission)
    st.write(mission_info[selected_mission])

    if selected_mission == "Global Space Mission Trends Over Time":
        missions_per_year = df['Year'].value_counts().sort_index()
        fig = px.line(
            x=missions_per_year.index, 
            y=missions_per_year.values, 
            title='Global Space Mission Trends Over Time'
        )
        fig.update_layout(
            xaxis_title='Year', 
            yaxis_title='Number of Space Missions'
        )
        st.plotly_chart(fig)

    elif selected_mission == "Distribution of Rocket Prices":
        fig = px.histogram(
            df,
            x="Price",
            title="Distribution of Rocket Prices",
            nbins=10,
            labels={"Price": "Rocket Price (Millions USD)"},
        )
        st.plotly_chart(fig)

    elif selected_mission == "Correlation analysis":
            # Compute the correlation matrix for relevant columns
            correlation_matrix = df[['Year', 'Price']].corr()

            # Plot the heatmap using matplotlib and seaborn
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
            ax.set_title('Correlation Matrix for Year and Price')

            # Display the heatmap in Streamlit
            st.pyplot(fig)

    elif selected_mission == "Mission Success Rate by Company":
        df_grouped = df.groupby(['Company', 'MissionStatus']).size().reset_index(name='Count')

        fig = px.bar(
            df_grouped,
            x="Company",
            y="Count",
            color="MissionStatus",
            title="Mission Success Rate by Company",
            labels={"MissionStatus": "Mission Status", "Company": "Company"},
            barmode="group",
            color_discrete_sequence=px.colors.sequential.Viridis  
        )

        fig.update_layout(
            xaxis={'categoryorder': 'total descending'}, 
            yaxis_title="Count",
            xaxis_title="Company",
            font=dict(size=12)
        )
        fig.update_layout(
            xaxis=dict(
                tickangle=45,  
                automargin=True,  
            ),
            width=1200,  
            height=600,  
            margin=dict(
                l=40,r=40,t=40,b=100
            ),
            title_font_size=16,
            xaxis_title="Company",
            yaxis_title="Count"
        )


        st.plotly_chart(fig)

    elif selected_mission == "Machine Learning":
        st.title("Machine Learning Analysis")
        st.markdown("### Predicting Mission Success Using Machine Learning")

        # Fill missing values in 'Price' with the median
        df['Price'] = df['Price'].fillna(df['Price'].median())

        # Drop rows with missing values in 'Time'
        df = df.dropna(subset=['Time'])

        # Drop unnecessary columns
        df = df.drop(columns=['Year', 'Country'])

        # Import required libraries
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, classification_report
        from sklearn.preprocessing import StandardScaler

        # One-Hot Encoding for categorical columns
        categorical_cols = ['Company', 'Location', 'Rocket', 'RocketStatus', 'Mission']
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        # Handle missing values (drop or impute)
        df_encoded.fillna(df_encoded.mean(), inplace=True)

        # Define Features (X) and Target (y)
        X = df_encoded.drop(columns=['MissionStatus'])  # Features
        y = df_encoded['MissionStatus']                 # Target

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize numerical features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Initialize the Random Forest Classifier
        rf = RandomForestClassifier(n_estimators=100, random_state=42)

        # Train the model
        rf.fit(X_train, y_train)

        # Make predictions
        y_pred = rf.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Display results in Streamlit
        st.subheader("Model Performance")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write("Classification Report:")
        st.dataframe(pd.DataFrame(report).transpose())


    elif selected_mission == "Top 10 Companies by Mission Count":
        company_counts = df['Company'].value_counts().reset_index()
        company_counts.columns = ['Company', 'Count']

        # Select top 10 companies by count
        top_10_companies = company_counts.head(10)

        fig = px.bar(
            top_10_companies,
            x='Company',
            y='Count',
            title='Top 10 Companies by Mission Count',
            labels={'Company': 'Company', 'Count': 'Count'}
        )

        fig.update_layout(
            xaxis=dict(
                tickangle=20,  
                automargin=True
            ),
            width=800,  
            height=500,  
            margin=dict(l=40, r=40, t=40, b=100),
            xaxis_title="Company",
            yaxis_title="Count",
            title_font_size=16
        )


        st.plotly_chart(fig)

    elif selected_mission == "Number of Missions by Country":
                # Extract country from the 'Location' column
        df['Country'] = df['Location'].str.split(',').str[2].str.strip()

        # Group by country and rocket status
        df_grouped = df.groupby(['Country', 'RocketStatus']).size().reset_index(name='Count')

        # Create the bar plot
        fig = px.bar(
            df_grouped,
            x="Country",
            y="Count",
            color="RocketStatus",
            title="Number of Missions by Country",
            labels={"Count": "Count", "RocketStatus": "Rocket Status"},
            barmode="group",
            color_discrete_sequence=px.colors.qualitative.Set1  
        )

        fig.update_layout(
            xaxis={'categoryorder': 'total descending'},  
            yaxis_title="Count",
            xaxis_title="Country",
            font=dict(size=12)
        )

        st.plotly_chart(fig)

    elif selected_mission == "Mission Trends and Rocket Prices Over Years":
        # Convert Year column to datetime format
        df["Year"] = pd.to_datetime(df["Year"], format="%Y", errors="coerce")

        # Drop rows where Year is invalid (if any)
        df = df.dropna(subset=["Year"])

        # Extract the year as a separate numeric column for analysis
        df["YearNumeric"] = df["Year"].dt.year

        # Sort data by the numeric Year column
        df = df.sort_values("YearNumeric")

        # Group data by YearNumeric and MissionStatus, taking the average Rocket Price
        df_aggregated = (
            df.groupby(["YearNumeric", "MissionStatus"])["Price"]
            .mean()
            .reset_index()
        )

        # Dynamically select tick values to reduce clutter
        unique_years = sorted(df_aggregated["YearNumeric"].unique())  # Get sorted unique years
        tickvals = unique_years[::max(1, len(unique_years) // 10)]  # Show approximately 10 ticks

        # Plot the data
        fig = px.line(
            df_aggregated,
            x="YearNumeric",
            y="Price",
            color="MissionStatus",
            title="Mission Trends and Rocket Prices Over Years",
            labels={"YearNumeric": "Year", "Price": "Rocket Price (Millions USD)", "MissionStatus": "Mission Status"},
            markers=True,
        )

        # Customize x-axis to reduce clutter
        fig.update_layout(
            xaxis=dict(
                tickmode="array",  # Use specific tick values
                tickvals=tickvals,  # Dynamically chosen ticks
                tickangle=45,  # Tilt the ticks for better readability
            ),
            width=900,
            height=600,
        )


        st.plotly_chart(fig)

# Gallery page
elif selection == "Gallery":
    st.title("Space Missions Gallery")
    st.markdown("### Explore breathtaking images from space missions")

    # List of image URLs
    images = [
        "https://live.staticflickr.com/65535/53645326260_29709f8917_b.jpg",
        "https://live.staticflickr.com/65535/53641212344_18e26f9784_b.jpg",
        "https://live.staticflickr.com/65535/53640965053_a76e8b0b1f_b.jpg",
        "https://live.staticflickr.com/65535/53772101548_ec27ed649d_b.jpg",
        "https://live.staticflickr.com/65535/53771717666_1327df2c55_b.jpg",
        "https://live.staticflickr.com/65535/53770816117_57d96a5d96_b.jpg",
        "https://live.staticflickr.com/65535/53988283029_1e4476523b_b.jpg"
    ]

    # Number of columns in the grid
    num_columns = 3
    columns = st.columns(num_columns)

    # Display images in a grid
    for idx, img_url in enumerate(images):
        col = columns[idx % num_columns]  # Choose the correct column
        with col:
            st.image(img_url, use_container_width=True)

# Feature page
elif selection == "Feature distribution analysis":
    st.title("Feature distribution analysis")
    st.markdown("Examine how individual features (variables) within our SpaceMission dataset are spread out or distributed. Select the feature from space missions.")

    favorite = st.multiselect(
        "Choose your feature:",
         ["Company", "Location", "Year", "MissionStatus", "Rockets",  "RocketStatus", "Price", "Mission"]
    )

    if not favorite:
        st.info("Please select at least one feature to analyze.")
    else:
        for feature in favorite:
            st.subheader(f"Analysis for {feature}")
            
            if feature == "Year":
                # Count occurrences of each year
                year_counts = df['Year'].value_counts().reset_index()
                year_counts.columns = ['Year', 'Count']  
                year_counts = year_counts.sort_values('Year')  # Sort by Year

                # Create a line chart using Plotly
                fig = px.line(
                    year_counts, 
                    x='Year', 
                    y='Count', 
                    title='Number of Records Over Years', 
                    markers=True
                )
                fig.update_layout(
                    xaxis_title='Year', 
                    yaxis_title='Count'
                )
                # Display the chart in Streamlit
                st.plotly_chart(fig)

            elif feature == "Location":
                # Scatter Geo plot for Launch Locations
                fig = px.scatter_geo(
                    df, 
                    locations='Location', 
                    locationmode='country names',
                    title='Space Mission Launch Locations'
                )
                st.plotly_chart(fig)

                location_success_rate = df.groupby('Location')['MissionStatus'].apply(lambda x: (x == 'Success').mean())

                fig2 = px.choropleth(locations=location_success_rate.index, locationmode='country names',
                                    color=location_success_rate.values, title='Mission Success Rate by Launch Location',
                                    labels={'color': 'Success Rate'})
                st.plotly_chart(fig2)

                # Calculate the top 10 locations
                top_locations = df['Location'].value_counts().head(10).reset_index()
                top_locations.columns = ['Location', 'Count']

                # Create a horizontal bar chart using Plotly
                fig = px.bar(
                    top_locations,
                    y='Location',
                    x='Count',
                    orientation='h',
                    title='Top 10 Launch Locations',
                    text='Count',
                    color='Count',
                    color_continuous_scale='plasma'
                )

                # Update layout
                fig.update_layout(yaxis_title='Location', xaxis_title='Count')

                # Display the chart in Streamlit
                st.plotly_chart(fig)

            elif feature == "Rockets":
                # Top 10 Rockets by count
                top_rockets = df['Rocket'].value_counts().head(10).reset_index()
                top_rockets.columns = ['Rocket', 'Count']  # Rename columns for clarity

                fig = px.bar(
                    top_rockets,
                    x='Count',
                    y='Rocket',
                    orientation='h',
                    title='Top 10 Rockets by Count',
                    text='Count',
                    color='Count',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(xaxis_title='Count', yaxis_title='Rocket')
                st.plotly_chart(fig)

            elif feature == "RocketStatus":
                # Group data by Year and RocketStatus
                rocket_status_year = df.groupby(['Year', 'RocketStatus']).size().reset_index(name='Count')

                # Create a stacked bar chart
                fig = px.bar(
                    rocket_status_year,
                    x='Year',
                    y='Count',
                    color='RocketStatus',
                    title='RocketStatus Distribution Over Years',
                    barmode='stack'
                )
                fig.update_layout(xaxis_title='Year', yaxis_title='Count')
                st.plotly_chart(fig)

            elif feature == "Price":
                # Box plot for Price distribution
                fig = px.box(
                    df,
                    y='Price',
                    title='Distribution of Prices',
                    points="all",
                    color_discrete_sequence=['green']
                )
                fig.update_layout(yaxis_title='Price')
                st.plotly_chart(fig)

            elif feature == "Mission": 
                # Combine all mission names into one string
                mission_words = ' '.join(df['Mission'].dropna().astype(str))

                # Generate the word cloud
                wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(mission_words)

                # Create a Matplotlib figure and display the word cloud
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                ax.set_title('Most Frequent Mission Names')
                st.pyplot(fig)


            elif feature == "Company":
                # Bar chart for top 10 companies by frequency
                top_companies = df['Company'].value_counts().head(10).reset_index()
                top_companies.columns = ['Company', 'Count']  # Rename columns for clarity

                fig = px.bar(
                    top_companies, 
                    x='Company', 
                    y='Count', 
                    title='Top 10 Companies by Count', 
                    text='Count',
                    color='Count', 
                    color_continuous_scale='viridis'
                )
                fig.update_layout(
                    xaxis_title='Company', 
                    yaxis_title='Count', 
                    xaxis={'categoryorder': 'total descending'}
                )
                st.plotly_chart(fig)

            elif feature == "MissionStatus":
                # Pie chart for MissionStatus occurrences
                mission_status_counts = df['MissionStatus'].value_counts().reset_index()
                mission_status_counts.columns = ['MissionStatus', 'Count']  

                fig = px.pie(
                    mission_status_counts, 
                    values='Count', 
                    names='MissionStatus', 
                    title='Proportion of Mission Statuses', 
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig)




# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Made with Streamlit** Explore the universe one mission at a time!"
)
