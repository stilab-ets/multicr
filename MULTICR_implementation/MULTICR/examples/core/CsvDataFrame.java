package core;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map.Entry;

public class CsvDataFrame extends DataFrame {
    
	private String seperator  ; 
	public CsvDataFrame(String sep) {
		super() ; 
		this.seperator = sep; 
	}

	@SuppressWarnings("finally")
	public void read_data_from_file(String path) throws FileNotFoundException {
		BufferedReader br = new BufferedReader(new FileReader(path));
		//reading columns i.e first line in the file
		try {
			String [] values ; 
			String line = "" ;
			Double value ; 
			HashMap<String,Object> min_max; 
			//reading columns i.e first line in the file
			line = br.readLine();
			String [] columns =  line.split(this.seperator) ; 
			// reading second line to initialize values
			line = br.readLine() ;
			//System.out.println(line) ; 
			values = line.split(seperator) ; 
			HashMap<String,Double> buffer = new HashMap<String,Double>() ; 
			for (int i = 0 ; i < values.length ; i++) {
				try {
					value = Double.parseDouble(values[i]) ;
					min_max = new HashMap<String,Object>() ; 
					
					
				    min_max.put("min",value) ; 
				    min_max.put("max",value) ; 
				    this.columns.put(columns[i],min_max) ;
				    buffer.put(columns[i], value);

				}
				finally {
					continue ; 
				}
				
			    
				
				
			    
				}
			this.getData().add(buffer) ; 
			while((line = br.readLine()) != null)
			{
				buffer = new HashMap<String,Double>() ; 
				
				values = line.split(this.seperator) ; 
				for (int i = 0 ; i < values.length ; i++) {
					try {
					value = Double.parseDouble(values[i]) ; 
					buffer.put(columns[i],value );
					
					min_max = this.columns.get(columns[i]) ; 
					min_max.put("min",Math.min(value, (double) min_max.get("min"))) ; 
					min_max.put("max",Math.max(value, (double) min_max.get("max"))) ; 
					
					}
					finally {
						continue ; 
					}
				}
				this.getData().add(buffer) ; 
				
			}
			br.close(); 
		} catch (IOException e) {
			e.printStackTrace();
		} 
	}
	public void print_data()
	{
		int i = 0 ; 
		for (HashMap<String,Double> row : this.getData())
		{
			System.out.println("i="+i++) ;
			/*for (Entry<String, Double> entry : row.entrySet())
			{
				System.out.println( "Key = " + entry.getKey() + 
                        ", Value = " + entry.getValue()); 
				System.out.println(this.columns.get(entry.getKey()).get("min")) ; 
				System.out.println(this.columns.get(entry.getKey()).get("max")) ; 
			}*/
			System.out.println(row.get("input missing")) ;
		}
	}
	public static void main(String[] arg)
	{
		String path = "/C:/Users/moataz/Desktop/MVGP/EGPC/DataFile/input_missing_data_with_header.csv" ; 
		String sep = "," ; 
		CsvDataFrame myData = new CsvDataFrame(sep) ; 
		try {
			myData.read_data_from_file(path);
			myData.print_data() ;
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
	}

}
