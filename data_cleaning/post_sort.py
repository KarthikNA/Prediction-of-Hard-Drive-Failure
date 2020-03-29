import pandas as pd
#bractrack n days if present and set the failure to 1 of a failed device.
#input: number of dats to bractrack : if n days - n entries of a drive will be set to 1.
#input :final_df -> dataframe to operate on
#returns changed data frame.
def set_failed_for_n_days(n,final_df):
    print("In setting ", n, "days to faiiled for all the failed devices")
    print(final_df.shape)
    #ensuring continous index is set
    final_df=final_df.reset_index(drop=True)

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


#drop rows based on how we want to consider good and bad drives:
#drop_good to TRUE will keep only n rows of each good data - extremely slow :(
#drop_bad to TRUE will remove all the bad entries that have failure as 0
#data_frame->data frame to be operated on
#n -> number of good entries to keep
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



#splitting into good and bad
def split_files(data_frame):    
    failed=data_frame['failure']==1
    failed_df=data_frame[failed]
    failed_df=failed_df.sort_values(by=['date'])
    failed_df.to_csv("only_failed.csv",index=False)

    good=data_frame['failure']==0
    good_df=data_frame[good]
    good_df=good_df.sort_values(by=['date'])
    good_df.to_csv("only_good.csv", index=False)




if __name__ == "__main__":
    
    #getting arguments  
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_loc", required=True,help="location of input csv file", action="store")
    parser.add_argument("--drop_good", required=False,help="Drop good entries and keep only n last entries",default=False, action="store_true")
    parser.add_argument("--drop_bad", required=False,help="Drop bad entries, where failure=0. Done after setting n last entries of failed drives to 1 ", default=False, action="store_true")
    parser.add_argument("--n", required=False,help="Number of days to back propagate", action="store", default=10)

    # Read arguments from the command line
    args = parser.parse_args()

    if args.file_loc:
        input_csv_file=args.file_loc

    print("Reading dataframe from ", input_csv_file)
    # data_frame=  pd.read_pickle("./k_sorted_q1.pkl")##this if reading from csv
    data_frame=pd.read_csv(input_csv_file) ##this if reading from csv
    print("Finished reading file")

    print("Number of days to backtrack for failed entries: ", args.n)
    failed_updated_df=set_failed_for_n_days(args.n,data_frame)

    print("Dropping the good entries? ", args.drop_good)
    print("Dropping the bad entries? ", args.drop_bad)
    final_df=dropping_rows(args.drop_good,args.drop_bad,failed_updated_df,args.n)
    print("After dropping rows",len(final_df))
    
    #Final CSV sorted by serial number and date
    final_df.to_csv('./final_q1.csv', index=False)
    print("Initial processing complete")

    #separating good and bad entries. Sorted by date only.
    split_files(final_df)
    print("Split files into good and bad")


