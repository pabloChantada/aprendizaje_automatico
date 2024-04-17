include("neural.jl")

using Pkg 
Pkg.add("XLSX") 
import XLSX 
xf = XLSX.readxlsx("p1//Dry_Bean_Dataset.xlsx")

dataset = xf["Dry_Beans_Dataset"] 
inputs = dataset[:, 1:16]
targets = dataset[:, 17]
targets = oneHotEncoding(targets)




XLSX.openxlsx("Dry_Bean_Dataset.xlsx", enable_cache=false) do f 
sheet = f["Sheet1"] 
for r in XLSX.eachrow(sheet) 

	# r is a `SheetRow`, values are read 
	# using column references 
	rn = XLSX.row_number(r) # `SheetRow` row number 
	v1 = r[1] # will read value at column 1 
	v2 = r[2]# will read value at column 2 
	v3 = r["B"] 
	v4 = r[3] 
	println("v1=$v1, v2=$v2, v3=$v3, v4=$v4") 
end 
end
