import java.util.ArrayList;
import java.util.List;

public class ListAddAllExamples {

    public static void main(String[] args) {
        List<Integer> val = new ArrayList<>();
        List<Integer> val1 = new ArrayList<>();
        List<Integer> val2 = new ArrayList<>();

        for (int i = 0; i < 10; i++) {
            val1.add(i);
            val2.add(i + 10);
        }

        val.addAll(val1);
        val.addAll(val2);


        for (Integer integer : val) {
            System.out.println(integer);
        }
    }
}
