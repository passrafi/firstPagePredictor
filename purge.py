#title,url,text,dead,by,score,time,type,id,parent,descendants,ranking,deleted,timestamp
#helper file that cleans corrupt data or incorrect data types


with open("data.csv") as f:
    with open("new.csv", "w") as f1:
        for line in f:
            row = line.split(',')
            #if no title, url and not a "story" type
            if len(row) < 6 or not row[0] or not row[1] or row[7] != "story":
                #skip row
                continue
            f1.write(line)
