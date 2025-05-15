package orcFileAnalysis;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.orc.*;
//import org.apache.orc.impl.BloomFilter;
//import org.apache.orc.impl.RecordReaderImpl;

import java.io.IOException;
import java.util.List;

public class ORCBloomFilter {
    public static void main(String[] args) {
        String orcFilePath = "path/to/your/orcfile.orc";

        Configuration conf = new Configuration();
        Path path = new Path(orcFilePath);

        try {
            // Create a reader for the ORC file
            Reader reader = OrcFile.createReader(path, OrcFile.readerOptions(conf));

            // Get the list of stripes in the ORC file
            List<StripeInformation> stripes = reader.getStripes();

            // Retrieve the schema of the ORC file
            TypeDescription schema = reader.getSchema();

            // Iterate through each stripe
            for (StripeInformation stripe : stripes) {

                RecordReader recordReader = reader.rows();
                //RecordReaderImpl recordReaderImpl = (RecordReaderImpl) recordReader;
                //StripeStatistics stripeStatistics = recordReaderImpl.getStripeStatistics();

                // Get the stripe statistics

                /*StripeStatistics stripeStatistics = ((RecordReaderImpl) reader.rowsOptions()
                        .range(stripe.getOffset(), stripe.getLength())
                        .build())
                        .getStripeStatistics();*/

                // Print the bloom filter for each column
                for (int col = 0; col < schema.getMaximumId(); col++) {
                    /*BloomFilter bloomFilter = stripeStatistics.getBloomFilter(col);
                    if (bloomFilter != null) {
                        System.out.println("Column ID: " + col);
                        System.out.println("Bloom Filter: " + bloomFilter.toString());
                    }*/
                }

            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
