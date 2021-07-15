import csv
 
# Open the input file 
input_file="f1.txt"
with open(input_file,"r") as input1:
    c1=csv.reader(input1)
    for r1 in c1:
	#print(r1)
        with open(r1[0], "r") as input2:
            #Set up CSV reader and process the header
            csvReader = csv.reader(input2)
            # header = next(csvReader)
            
          
            # Make an empty list
            coordList= []
            # coordList1 = []
            # coordList2 = []
            # coordList3 = []
            # coordList4 = []
            # coordList5 = []
            # coordList6 = []
            # coordList7 = []
            
            # coordList.append(r1[0])
            # flag=0
            # Loop through the lines in the file and get each coordinate
            for row in csvReader:
                # print(row)
                try:
                    if row[0].index('bucket 0') >=0 :
                        # print(row.index('Trial Time:'))
                        b0=float(row[0][8:])
                        coordList0.append(b1)
                    
                except ValueError:
                    n=1
                try:
                    if row[0].index('bucket 1') >=0 :
                        # print(row.index('Trial Time:'))
                        b1=float(row[0][8:])
                        coordList1.append(b1)
                    
                except ValueError:
                    n=1
                try:
                    if row[0].index('bucket 2') >=0 :
                        # print(row.index('Trial Time:'))
                        b2=float(row[0][8:])
                        coordList2.append(b2)
                    
                except ValueError:
                    n=1
                try:
                    if row[0].index('bucket 3') >=0 :
                        # print(row.index('Trial Time:'))
                        b3=float(row[0][8:])
                        coordList3.append(b3)
                    
                except ValueError:
                    n=1
                try:
                    if row[0].index('bucket 4') >=0 :
                        # print(row.index('Trial Time:'))
                        b4=float(row[0][8:])
                        coordList4.append(b4)
                    
                except ValueError:
                    n=1
                try:
                    if row[0].index('bucket 5') >=0 :
                        # print(row.index('Trial Time:'))
                        b5=float(row[0][8:])
                        coordList5.append(b5)
                    
                except ValueError:
                    n=1
                try:
                    if row[0].index('bucket 6') >=0 :
                        # print(row.index('Trial Time:'))
                        b6=float(row[0][8:])
                        coordList6.append(b6)
                    
                except ValueError:
                    n=1
                try:
                    if row[0].index('bucket 7') >=0 :
                        # print(row.index('Trial Time:'))
                        b7=float(row[0][8:])
                        coordList7.append(b7)
                    
                except ValueError:
                    n=1
            print(coordList0)
            print(coordList1)
            print(coordList2)
            print(coordList3)
            print(coordList4)
            print(coordList5)
            print(coordList6)

            print(coordList7)
