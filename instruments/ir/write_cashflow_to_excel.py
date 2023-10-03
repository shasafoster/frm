import xlsxwriter

def write_trade(path,cashflow_schedules,DV01):

    def write_leg(df, row, dynamic=True):
        df = df.dropna()
        
        # Write the header of the cashflow dataframe
        col_names = df.columns.values
        for c,v in enumerate(col_names):
            worksheet.write(row,c,v,header_format)
        row +=  1
        
        # Write the cashflow dataframe values
        arr = df.values
        nb_rows = len(arr)
        nb_cols = len(arr[0])
        map_colnb_to_colname = {i : col_name for i,col_name in enumerate(col_names)}
        worksheet.set_column(0, 12, 14) # Set column width of colums 0-12 to 14    
        
        # Use formula's for the net floating rate, cash flow & cash flow PV columns
        if dynamic: 
            hardcoded_cols = {'Start Date', 'End Date', 'Payment Date', 'Days', 'Year Fraction', 'Notional', 'Fixed Rate', 'Fixing Date', 'Index Rate','Spread','Discount Factor'}
            for r in range(nb_rows):
                for c,col_name in enumerate(col_names):
                    if col_name in hardcoded_cols:
                        worksheet.write(row,c,arr[r,c],format_dict[map_colnb_to_colname[c]])
                    else:
                        if nb_cols == 10:
                            if col_name == 'Cash Flow':
                                worksheet.write(row,c,'=E' + str(row+1)+'*F'+str(row+1)+'*G'+str(row+1),format_dict[map_colnb_to_colname[c]])
                            elif col_name == 'Cash Flow PV':
                                worksheet.write(row,c,'=H' + str(row+1)+'*I'+str(row+1),format_dict[map_colnb_to_colname[c]])
                            else:
                                raise ValueError('Invalid value: ', col_name, nb_cols)
                        elif nb_cols == 13:
                            if col_name == 'Floating Rate':
                                worksheet.write(row,c,'=H' + str(row+1)+'+I'+str(row+1),format_dict[map_colnb_to_colname[c]])
                            elif col_name == 'Cash Flow':
                                worksheet.write(row,c,'=E' + str(row+1)+'*F'+str(row+1)+'*J'+str(row+1),format_dict[map_colnb_to_colname[c]])
                            elif col_name == 'Cash Flow PV':
                                worksheet.write(row,c,'=K' + str(row+1)+'*L'+str(row+1),format_dict[map_colnb_to_colname[c]])
                            else:
                                raise ValueError('Invalid value: ', col_name, nb_cols)
                row +=  1
                                
        # Hardcode data  
        else:
            for r in range(nb_rows):
                for c in range(nb_cols):
                    worksheet.write(row,c,arr[r,c],format_dict[map_colnb_to_colname[c]])
                row +=  1
        
        # Write formula to sum the discounted cashflows
        col_letter = str(xlsxwriter.utility.xl_col_to_name(nb_cols-1))
        worksheet.write(row,nb_cols-1,'=SUM(' + col_letter + str(row-nb_rows+1) + ':' + col_letter + str(row) + ')',sum_format)
        cell = str(xlsxwriter.utility.xl_col_to_name(nb_cols-1)) + str(row+1)
        row += 2
        
        return row, cell
    
    
    def write_cashflows(worksheet, dynamic_flag):
        row = 0
        cells = []
        for i,leg in enumerate(cashflow_schedules):
            row,cell = write_leg(cashflow_schedules[i], row, dynamic_flag)
            cells.append(cell)
            
        worksheet.write(row,1,"MV",header_format)
        worksheet.write(row,2,"DV01",header_format)
        row += 1
        for i,cell in enumerate(cells):
            worksheet.write(row,0,"LEG"+str(i+1),number_format)
            worksheet.write(row,1,"="+cell,number_format)
            worksheet.write(row,2,DV01[i],number_format)
            row += 1
            
        worksheet.write(row,0,"Net")
        worksheet.write(row,1,"=SUM(" + "B" + str(row - len(cells) + 1) + ":B" + str(row) + ')',sum_format)
        worksheet.write(row,2,"=SUM(" + "C" + str(row - len(cells) + 1) + ":C" + str(row) + ')',sum_format)
    
        
    workbook = xlsxwriter.Workbook(path)
    
    date_format = workbook.add_format({'num_format': 'd-mmm-yyyy;;'})
    number_format = workbook.add_format({'num_format': '#,##0;(#,##0)'})
    fraction_format = workbook.add_format({'num_format': '0.0000_-'})
    percentage_format = workbook.add_format({'num_format': '0.0000%'})
    general_format = workbook.add_format({'num_format': 'General'})
    header_format = workbook.add_format()
    header_format.set_bottom()
    sum_format = workbook.add_format({'num_format': '#,##0;(#,##0)'})
    sum_format.set_top()
    sum_format.set_bottom()
    
    format_dict = {'Fixing Date' : date_format,
                   'Start Date' : date_format,
                   'End Date' : date_format,
                   'Payment Date' : date_format,
                   'Days' : general_format,
                   'Year Fraction' : fraction_format,
                   'Notional' : number_format,
                   'Fixed Rate' : percentage_format,
                   'Index Rate' : percentage_format,
                   'Spread' : percentage_format,
                   'Floating Rate' : percentage_format,
                   'Cash Flow' : number_format,
                   'Discount Factor' : fraction_format,
                   'Cash Flow PV' : number_format}
    
    worksheet = workbook.add_worksheet('Hardcoded')
    write_cashflows(worksheet, dynamic_flag=False)
    worksheet = workbook.add_worksheet('Dynamic')
    write_cashflows(worksheet, dynamic_flag=True)
    
    workbook.close()
    

