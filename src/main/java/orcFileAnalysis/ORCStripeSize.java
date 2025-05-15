package orcFileAnalysis;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.orc.*;

import java.io.IOException;
import java.util.List;

public class ORCStripeSize {

    public static void main(String[] args) {
        //String orcFilePath = "/Downloads/part-00000-aa922f97-016b-4b65-beec-4ad5bc41f5fb-c000.snappy.orc";
        String orcFilePath = "/Downloads/part-00150-14fede91-8108-4a24-aeac-4492ad376dbf-c000.zstd.orc";

        Configuration conf = new Configuration();
        conf.set("fs.defaultFS", "file:///");
        Path path = new Path(orcFilePath);

        try {
            // Create a reader for the ORC file
            Reader reader = OrcFile.createReader(path, OrcFile.readerOptions(conf));

            TypeDescription schema = reader.getSchema();

            // Get the list of stripes in the ORC file
            List<StripeInformation> stripes = reader.getStripes();

            // Iterate through each stripe and print its size
            for (StripeInformation stripe : stripes) {

                //RecordReader recordReader = reader.rows();
                //recordReader.seekToRow(0);

                //OrcProto.BloomFilterIndex[] bloomFilterIndices = reader.get();

                System.out.println("Stripe offset: " + stripe.getOffset());
                System.out.println("Stripe length: " + stripe.getLength());
            }

            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
