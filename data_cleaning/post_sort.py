import pandas as pd
def set_failed_for_n_days(n,final_df):
    print("In setting ", n, "days to faiiled for all the failed devices")
    #get all indices where the failure is 1, so we could
    #back track n days and set entries to 1 and remove the rest
    failure_indices=final_df.loc[final_df['failure']==1].index
    #needed to remove the extra entries
    failed_serial_numbers=[]

    for i in failure_indices:
        current_serial_number=final_df.iloc[i]['serial_number']
        failed_serial_numbers.append(current_serial_number)
        print("Updating for serial number:", current_serial_number)
        #we check for n rows above the failed state and set to failed if the serial number matches
        final_df['failure'].loc[i-n+1:i+1] = ((((final_df['serial_number']==current_serial_number) | (final_df['failure']==1))*1).loc[i-n+1:i+1])
    print("Total failed scenarios - number of entries where failuer=1",len(final_df.loc[final_df['failure']==1].index))    
    print("Total failed serial numbers:",len(failed_serial_numbers))
    print("Total entries: both good and bad ", len(final_df))
    print("Failure update completed")
    return final_df

# def get_failed_serial_numbers(failure_indices):
#     failed_serial_numbers=set()
#     for i in failure_indices:
#         failed_serial_numbers.update([final_df.iloc[i]['serial_number']])
#     return failed_serial_numbers


def dropping_rows(drop_good,drop_bad, data_frame,n=10):
# #drop the rows which have failed serial number, but has failure as 0 
    final_df=data_frame
    failure_indices=final_df.loc[final_df['failure']==1].index
    failed_serial_numbers=set()
    for i in failure_indices:
        failed_serial_numbers.update([final_df.iloc[i]['serial_number']])
        
    # failed_serial_numbers=get_failed_serial_numbers(failure_indices)
    
   
    print("Failed serial numbers: ", failed_serial_numbers)
    if(drop_bad):
        print("Removing unfailed entries of failed serial numbers")
        for i in failed_serial_numbers:
            to_drop=final_df[(final_df['serial_number']==i )& ~(final_df['failure']==1)]
            final_df=final_df.drop(to_drop.index)
    if(drop_good):
        all_serial_numbers=set(final_df['serial_number'].unique())
        good_serial_numbers=all_serial_numbers.difference(failed_serial_numbers)
        for i in good_serial_numbers:
            filtered_frame=final_df.loc[final_df['serial_number']==i]
            number_to_drop=len(filtered_frame)-n
            to_drop=final_df.loc[final_df['serial_number']==i].head(number_to_drop)
            final_df=final_df.drop(to_drop.index)
    return final_df


print("Read dataframe")
data_frame=pd.read_pickle("./sorted_q1.pkl")
# data_frame=pd.read_csv("<csv file>")

failed_updated_df=set_failed_for_n_days(10,data_frame)            
final_df=dropping_rows(False,False,failed_updated_df,10)
print("After, drop",len(final_df))
#Final Pickel
final_df.to_pickle('./final_file_q1_false_false.pkl')
print("Initial processing complete")