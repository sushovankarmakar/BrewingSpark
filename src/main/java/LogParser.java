import java.io.*;

public class LogParser {

    public static void main(String[] args) throws IOException {
        // File path is passed as parameter
        File file = new File("/Downloads/abc.log");
        // Creating an object of BufferedReader class
        BufferedReader br = new BufferedReader(new FileReader(file));

        FileWriter fw = new FileWriter("extracted.txt");

        // Declaring a string variable
        String st;
        // Condition holds true till
        // there is character in a string
        while ((st = br.readLine()) != null) {

            if (st.contains("SUSHOVAN")) {
                System.out.println(st);
                fw.write(st + "\n");
            }
        }

        br.close();
        fw.close();
    }
}
