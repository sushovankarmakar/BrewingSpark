package orcFileAnalysis;

import org.apache.orc.*;
//import org.apache.orc.impl.ReaderImpl;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;

import java.io.IOException;

public class OrcFileInspector {

    public static void main(String[] args) throws IOException {
        // ORC file path
        String orcFilePath = "/data/poc_output/part-00150-14fede91-8108-4a24-aeac-4492ad376dbf-c000.zstd.orc";

        // Hadoop Configuration
        Configuration conf = new Configuration();

        // ORC Reader
        Reader reader = OrcFile.createReader(new Path(orcFilePath), OrcFile.readerOptions(conf));

        // Print Header
        printHeader(reader);

        // Print Footer
        printFooter(reader);

        // Print Postscript
        printPostscript(reader);

        // Print Stripes Information
        printStripes(reader);
    }

    private static void printHeader(Reader reader) {
        System.out.println("== Header ==");
        System.out.println("File Type: " + reader.getFileVersion().getName());
    }

    private static void printFooter(Reader reader) throws IOException {
        System.out.println("== Footer ==");
        System.out.println("Number of Rows: " + reader.getNumberOfRows());
        System.out.println("Compression: " + reader.getCompressionKind());
        System.out.println("Compression Size: " + reader.getCompressionSize());
        System.out.println("Schema: " + reader.getSchema());
        System.out.println("File Length: " + reader.getContentLength());
    }

    private static void printPostscript(Reader reader) throws IOException {
        System.out.println("== Postscript ==");
        /*OrcProto.PostScript postscript = ((ReaderImpl) reader).getPostScript();
        System.out.println("Footer Length: " + postscript.getFooterLength());
        System.out.println("Compression: " + postscript.getCompression());
        System.out.println("Compression Block Size: " + postscript.getCompressionBlockSize());
        System.out.println("Version List: " + postscript.getVersionList());*/
    }

    private static void printStripes(Reader reader) throws IOException {
        System.out.println("== Stripes ==");
        for (StripeInformation stripe : reader.getStripes()) {
            System.out.println(stripe.getLength());
            System.out.println("Stripe Offset: " + stripe.getOffset());
            System.out.println("Number of Rows: " + stripe.getNumberOfRows());
            System.out.println("Data Length: " + stripe.getDataLength());
            System.out.println("Index Length: " + stripe.getIndexLength());
            System.out.println("Footer Length: " + stripe.getFooterLength());
            System.out.println();
        }
    }
}
