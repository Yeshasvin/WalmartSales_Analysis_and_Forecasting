
import matplotlib.pyplot as plt

class SalesPlots:
    def __init__(self, df):
        self.df = df
    

    #-------------------------------------------------------------------------------------------
    # Function to plot Feautre vs Weekly Sales Graphs
    #-------------------------------------------------------------------------------------------


    def feature_sales_plot(self, store_number: int):
        

        store_df = self.df[self.df['Store'] == store_number]

        fig, axs = plt.subplots(2,2, figsize=(15,10))
        
        print(f"Features vs Weekly Sales plots for store {store_number}")

        #-------------------------------------------------------------------------------------------
        # Graph no.1
        #-------------------------------------------------------------------------------------------

        axs[0,0].plot(store_df.index, store_df['Weekly_Sales'], label='Sales', color='blue')
        axs[0,0].set_ylabel('Weekly Sales')
        axs[0,0].legend(loc='upper left')
        axs[0,0].tick_params(axis='x', rotation=45)

        axs1 = axs[0,0].twinx()
        axs1.plot(store_df.index, store_df['Temperature'], label='Temperature', color='red')
        axs1.set_ylabel('Temperature')
        axs1.legend(loc='upper right')

        #-------------------------------------------------------------------------------------------
        # Graph no.2
        #-------------------------------------------------------------------------------------------

        axs[0,1].plot(store_df.index, store_df['Weekly_Sales'], label='Sales', color='blue')
        axs[0,1].set_ylabel('Weekly Sales')
        axs[0,1].legend(loc='upper left')
        axs[0,1].tick_params(axis='x', rotation=45)

        axs2= axs[0,1].twinx()
        axs2.plot(store_df.index, store_df['Unemployment'], label='Unemployment', color='red')
        axs2.set_ylabel('Unemployment')
        axs2.legend(loc='upper right')

        #-------------------------------------------------------------------------------------------
        # Graph no.3
        #-------------------------------------------------------------------------------------------

        axs[1,0].plot(store_df.index, store_df['Weekly_Sales'], label='Sales', color='blue')
        axs[1,0].set_ylabel('Weekly Sales')
        axs[1,0].legend(loc='upper left')
        axs[1,0].tick_params(axis='x', rotation=45)

        axs3 = axs[1,0].twinx()

        axs3.plot(store_df.index, store_df['Fuel_Price'], label= 'Fuel Price', color='red')
        axs3.set_ylabel('Fuel_Price')
        axs3.legend(loc='upper right')

        #-------------------------------------------------------------------------------------------
        # Graph no.4
        #-------------------------------------------------------------------------------------------

        axs[1,1].plot(store_df.index, store_df['Weekly_Sales'], label='Sales', color='blue')
        axs[1,1].set_ylabel('Weekly Sales')
        axs[1,1].legend(loc='upper left')
        axs[1,1].tick_params(axis='x', rotation=45)

        axs4 = axs[1,1].twinx()

        axs4.plot(store_df.index, store_df['CPI'], label='CPI', color='red')
        axs4.set_ylabel('CPI')
        axs4.legend(loc='upper right')



        plt.tight_layout()
        plt.plot()
        
        return store_df
        
    #-------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------


    #-------------------------------------------------------------------------------------------
    # Function to plot Custom Feautre vs Weekly Sales Graphs
    #-------------------------------------------------------------------------------------------

    def custom_feature_sales_plot(self, store_number, main_col: str, subsidiary_cols: list):
    # Filter DataFrame for the selected store
        store_df = self.df[self.df['Store'] == store_number].copy()
        
        # Number of required plots based on subsidiary columns
        num_plots = len(subsidiary_cols)
        rows = (num_plots // 2) + (num_plots % 2)  # Calculate rows for subplots

        # Create subplots
        fig, axs = plt.subplots(rows, 2, figsize=(15, rows * 5))
        axs = axs.ravel()  # Flatten the 2D array of axes for easier iteration

        print(f"Main vs Subsidiary Columns plots for store {store_number}")

        # Loop through subsidiary columns and create subplots
        for i, sub_col in enumerate(subsidiary_cols):
            if sub_col not in store_df.columns:
                print(f"Column '{sub_col}' does not exist in the DataFrame.")
                continue

            # Plot main column (Weekly Sales)
            axs[i].plot(store_df.index, store_df[main_col], label='Weekly Sales', color='blue')
            axs[i].set_ylabel('Weekly Sales')
            axs[i].set_title(f'Weekly Sales vs {sub_col}')  # Add title for clarity
            axs[i].legend(loc='upper left')
            axs[i].tick_params(axis='x', rotation=45)

            # Create a twin axis for the subsidiary column
            axs1 = axs[i].twinx()
            axs1.plot(store_df.index, store_df[sub_col], label=sub_col, color='red')
            axs1.set_ylabel(sub_col)
            axs1.legend(loc='upper right')

        # Hide any unused axes (if any)
        for j in range(i + 1, len(axs)):  # Hide any remaining axes
            axs[j].axis('off')

        # Adjust layout and display plots
        plt.tight_layout()
        plt.show()
        
        return store_df

    #-------------------------------------------------------------------------------------------
    # Function to plot Aggregate Sales 
    #-------------------------------------------------------------------------------------------

    def aggregate_sales_plot(self, store_number: int):
        

        store_df = self.df[self.df['Store'] == store_number].copy()
        
        store_df['Month'] = store_df.index.month
        store_df['Year'] = store_df.index.year
        
        print(f"Features vs Weekly Sales plots for store {store_number}")

        agg_set = ['mean', 'max', 'min']
        
        print(f'Aggregate Sales Plots for store {store_number}')
        
        #-------------------------------------------------------------------------------------------
        # Plotting Aggregate plots
        #-------------------------------------------------------------------------------------------

        fig, axs = plt.subplots(3,2, figsize=(15,15))

        #-------------------------------------------------------------------------------------------
        # Graph no.1
        #-------------------------------------------------------------------------------------------

        store_df.groupby(['Holiday_Flag'])['Weekly_Sales'].agg(agg_set).sort_values(by=agg_set, ascending=False).plot(kind='bar',
                                                                                                                color= ['b', 'g', 'r'],
                                                                                                                ax = axs[0,0])
        axs[0,0].set_title('Aggregate Weekly Sales W.R.T  Holiday Flag')
        axs[0,0].tick_params(axis='x', rotation=0)


        #-------------------------------------------------------------------------------------------
        # Graph no.2
        #-------------------------------------------------------------------------------------------

        store_df.groupby(['Holiday_Flag'])['Weekly_Sales'].sum().sort_values(ascending=False).plot(kind='bar',
                                                                                                            color= ['b', 'g', 'r'],
                                                                                                            ax = axs[0,1])
        axs[0,1].set_title('Sum of  Weekly Sales W.R.T  Holiday Flag')
        axs[0,1].tick_params(axis='x', rotation=0)

        #-------------------------------------------------------------------------------------------
        # Graph no.3
        #-------------------------------------------------------------------------------------------
        store_df.groupby('Year')['Weekly_Sales'].agg(agg_set).sort_values(by=agg_set, ascending=False).plot(kind='bar',
                                                                                                        color= ['r', 'g', 'b'],
                                                                                                        ax=axs[1,0])
        axs[1,0].set_title('Year-wise mean, max, min for Store 2')
        axs[1,0].tick_params(axis='x', rotation=0)


        #-------------------------------------------------------------------------------------------
        # Graph no.4
        #-------------------------------------------------------------------------------------------

        store_df.groupby('Year')['Weekly_Sales'].mean().sort_values(ascending=False).plot(kind='bar',
                                                                                        color= ['r', 'g', 'b'],
                                                                                        ax=axs[1,1])
        axs[1,1].set_title('Year-wise Sum for Store 2 ')
        axs[1,1].tick_params(axis='x', rotation=0)


        #-------------------------------------------------------------------------------------------
        # Graph no.5
        #-------------------------------------------------------------------------------------------

        store_df.groupby('Month')['Weekly_Sales'].agg(agg_set).sort_values(by=agg_set, ascending=False).plot(kind='bar',
                                                                                                            color= ['r', 'g', 'b'],
                                                                                                            ax=axs[2,0])
        axs[2,0].set_title('Month-wise mean, max, min for Store 2')
        axs[2,0].tick_params(axis='x', rotation=0)

        #-------------------------------------------------------------------------------------------
        # Graph no.6
        #-------------------------------------------------------------------------------------------

        store_df.groupby('Month')['Weekly_Sales'].mean().sort_values(ascending=False).plot(kind='bar',
                                                                                        color= ['r', 'g', 'b'],
                                                                                        ax=axs[2,1])
        axs[2,1].set_title('Month-wise Sum for Store 2 ')
        axs[2,1].tick_params(axis='x', rotation=0)


        plt.tight_layout()

        
        return store_df
        
    #-------------------------------------------------------------------------------------------
    # Function to plot Aggregate Sales plot  and Feature sales plot Together 
    #-------------------------------------------------------------------------------------------

    def combined_plots(self, store_number: int):
        self.feature_sales_plot(store_number)
        self.aggregate_sales_plot(store_number)
        
        
        
        
    #-------------------------------------------------------------------------------------------
    # Function to plot Feature vs  Sales in a given year
    #-------------------------------------------------------------------------------------------        
        
    def feature_sales_plot_year_wise(self, store_number: int, year: int):
        

        store_df = self.df[self.df['Store'] == store_number].copy()
        store_df = store_df[store_df.index.year == year]

        fig, axs = plt.subplots(2,2, figsize=(15,10))
        
        print(f"Features vs Weekly Sales plots for store {store_number} in the Year {year}")

        #-------------------------------------------------------------------------------------------
        # Graph no.1
        #-------------------------------------------------------------------------------------------

        axs[0,0].plot(store_df.index, store_df['Weekly_Sales'], label='Sales', color='blue')
        axs[0,0].set_ylabel('Weekly Sales')
        axs[0,0].legend(loc='upper left')
        axs[0,0].tick_params(axis='x', rotation=45)

        axs1 = axs[0,0].twinx()
        axs1.plot(store_df.index, store_df['Temperature'], label='Temperature', color='red')
        axs1.set_ylabel('Temperature')
        axs1.legend(loc='upper right')

        #-------------------------------------------------------------------------------------------
        # Graph no.2
        #-------------------------------------------------------------------------------------------

        axs[0,1].plot(store_df.index, store_df['Weekly_Sales'], label='Sales', color='blue')
        axs[0,1].set_ylabel('Weekly Sales')
        axs[0,1].legend(loc='upper left')
        axs[0,1].tick_params(axis='x', rotation=45)

        axs2= axs[0,1].twinx()
        axs2.plot(store_df.index, store_df['Unemployment'], label='Unemployment', color='red')
        axs2.set_ylabel('Unemployment')
        axs2.legend(loc='upper right')

        #-------------------------------------------------------------------------------------------
        # Graph no.3
        #-------------------------------------------------------------------------------------------

        axs[1,0].plot(store_df.index, store_df['Weekly_Sales'], label='Sales', color='blue')
        axs[1,0].set_ylabel('Weekly Sales')
        axs[1,0].legend(loc='upper left')
        axs[1,0].tick_params(axis='x', rotation=45)

        axs3 = axs[1,0].twinx()

        axs3.plot(store_df.index, store_df['Fuel_Price'], label= 'Fuel Price', color='red')
        axs3.set_ylabel('Fuel_Price')
        axs3.legend(loc='upper right')

        #-------------------------------------------------------------------------------------------
        # Graph no.4
        #-------------------------------------------------------------------------------------------

        axs[1,1].plot(store_df.index, store_df['Weekly_Sales'], label='Sales', color='blue')
        axs[1,1].set_ylabel('Weekly Sales')
        axs[1,1].legend(loc='upper left')
        axs[1,1].tick_params(axis='x', rotation=45)

        axs4 = axs[1,1].twinx()

        axs4.plot(store_df.index, store_df['CPI'], label='CPI', color='red')
        axs4.set_ylabel('CPI')
        axs4.legend(loc='upper right')



        plt.tight_layout()
        plt.plot()
        
        return store_df
        
    #-------------------------------------------------------------------------------------------
    # Function to plot Custom Feautre vs Weekly Sales Graphs
    #-------------------------------------------------------------------------------------------

    def custom_feature_sales_plot_year_wise(self, store_number, main_col: str, subsidiary_cols: list, year:int):
    # Filter DataFrame for the selected store
        store_df = self.df[self.df['Store'] == store_number].copy()
        store_df = store_df[store_df.index.year == year]
        
        # Number of required plots based on subsidiary columns
        num_plots = len(subsidiary_cols)
        rows = (num_plots // 2) + (num_plots % 2)  # Calculate rows for subplots

        # Create subplots
        fig, axs = plt.subplots(rows, 2, figsize=(15, rows * 5))
        axs = axs.ravel()  # Flatten the 2D array of axes for easier iteration

        print(f"Main vs Subsidiary Columns plots for store {store_number}")

        # Loop through subsidiary columns and create subplots
        for i, sub_col in enumerate(subsidiary_cols):
            if sub_col not in store_df.columns:
                print(f"Column '{sub_col}' does not exist in the DataFrame.")
                continue

            # Plot main column (Weekly Sales)
            axs[i].plot(store_df.index, store_df[main_col], label='Weekly Sales', color='blue')
            axs[i].set_ylabel('Weekly Sales')
            axs[i].set_title(f'Weekly Sales vs {sub_col}')  # Add title for clarity
            axs[i].legend(loc='upper left')
            axs[i].tick_params(axis='x', rotation=45)

            # Create a twin axis for the subsidiary column
            axs1 = axs[i].twinx()
            axs1.plot(store_df.index, store_df[sub_col], label=sub_col, color='red')
            axs1.set_ylabel(sub_col)
            axs1.legend(loc='upper right')

        # Hide any unused axes (if any)
        for j in range(i + 1, len(axs)):  # Hide any remaining axes
            axs[j].axis('off')

        # Adjust layout and display plots
        plt.tight_layout()

        
        return store_df        
    

        
        
        

    #-------------------------------------------------------------------------------------------
    # Function to plot Aggregate Sales in a given year
    #-------------------------------------------------------------------------------------------        
        
            
    def aggregate_sales_plot_year_wise(self, store_number: int, year: int):
        

        store_df = self.df[self.df['Store'] == store_number].copy()
        store_df = store_df[store_df.index.year == year]
        
        
        store_df['Month'] = store_df.index.month
        store_df['Year'] = store_df.index.year
        
        print(f"Features vs Weekly Sales plots for store {store_number}")

        agg_set = ['mean', 'max', 'min']
        
        print(f'Aggregate Sales Plots for store {store_number} in the Year {year}')
        
        #-------------------------------------------------------------------------------------------
        # Plotting Aggregate plots
        #-------------------------------------------------------------------------------------------

        fig, axs = plt.subplots(3,2, figsize=(15,15))

        #-------------------------------------------------------------------------------------------
        # Graph no.1
        #-------------------------------------------------------------------------------------------

        store_df.groupby(['Holiday_Flag'])['Weekly_Sales'].agg(agg_set).sort_values(by=agg_set, ascending=False).plot(kind='bar',
                                                                                                                color= ['b', 'g', 'r'],
                                                                                                                ax = axs[0,0])
        axs[0,0].set_title('Aggregate Weekly Sales W.R.T  Holiday Flag')
        axs[0,0].tick_params(axis='x', rotation=0)


        #-------------------------------------------------------------------------------------------
        # Graph no.2
        #-------------------------------------------------------------------------------------------

        store_df.groupby(['Holiday_Flag'])['Weekly_Sales'].sum().sort_values(ascending=False).plot(kind='bar',
                                                                                                            color= ['b', 'g', 'r'],
                                                                                                            ax = axs[0,1])
        axs[0,1].set_title('Sum of  Weekly Sales W.R.T  Holiday Flag')
        axs[0,1].tick_params(axis='x', rotation=0)

        #-------------------------------------------------------------------------------------------
        # Graph no.3
        #-------------------------------------------------------------------------------------------
        store_df.groupby('Year')['Weekly_Sales'].agg(agg_set).sort_values(by=agg_set, ascending=False).plot(kind='bar',
                                                                                                        color= ['r', 'g', 'b'],
                                                                                                        ax=axs[1,0])
        axs[1,0].set_title('Year-wise mean, max, min for Store 2')
        axs[1,0].tick_params(axis='x', rotation=0)


        #-------------------------------------------------------------------------------------------
        # Graph no.4
        #-------------------------------------------------------------------------------------------

        store_df.groupby('Year')['Weekly_Sales'].mean().sort_values(ascending=False).plot(kind='bar',
                                                                                        color= ['r', 'g', 'b'],
                                                                                        ax=axs[1,1])
        axs[1,1].set_title('Year-wise Sum for Store 2 ')
        axs[1,1].tick_params(axis='x', rotation=0)


        #-------------------------------------------------------------------------------------------
        # Graph no.5
        #-------------------------------------------------------------------------------------------

        store_df.groupby('Month')['Weekly_Sales'].agg(agg_set).sort_values(by=agg_set, ascending=False).plot(kind='bar',
                                                                                                            color= ['r', 'g', 'b'],
                                                                                                            ax=axs[2,0])
        axs[2,0].set_title('Month-wise mean, max, min for Store 2')
        axs[2,0].tick_params(axis='x', rotation=0)

        #-------------------------------------------------------------------------------------------
        # Graph no.6
        #-------------------------------------------------------------------------------------------

        store_df.groupby('Month')['Weekly_Sales'].mean().sort_values(ascending=False).plot(kind='bar',
                                                                                        color= ['r', 'g', 'b'],
                                                                                        ax=axs[2,1])
        axs[2,1].set_title('Month-wise Sum for Store 2 ')
        axs[2,1].tick_params(axis='x', rotation=0)


        plt.tight_layout()
    
        
        return store_df
        
    #-------------------------------------------------------------------------------------------
    # Function to plot Aggregate Sales  and Feature vs Sales in a given year
    #-------------------------------------------------------------------------------------------        
    
    def combined_plots_year_wise(self, store_number: int, year: int):
        self.feature_sales_plot_year_wise(store_number, year)
        self.aggregate_sales_plot_year_wise(store_number, year)
    