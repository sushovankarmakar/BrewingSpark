import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

public class MapInternals {

    public static void main(String[] args) {

        Map<String, Map<String, Integer>> outerMap = new HashMap<>();

        Map<String, Integer> innerMap = new TreeMap<>();
        innerMap.put("10_June", 0);

        outerMap.put("123", innerMap);

        String[] outerKeys = {"123"};

        Map<String, Integer> innerMap1 = new TreeMap<>();
        for (String outerKey : outerKeys) {

            if (outerMap.containsKey(outerKey)) {
                Map<String, Integer> innerMap2 = outerMap.get(outerKey);
                innerMap2.put("11_June", 1);

                //outerMap.put(outerKey, innerMap2);
            }
        }

        for (Map.Entry<String, Map<String, Integer>> entry : outerMap.entrySet()) {
            System.out.println(entry.getKey());
            System.out.println(entry.getValue());
        }
    }
}
