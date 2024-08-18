package data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class DataReader {

    private final int rows = 28;
    private final int columns = 28;

    public List<Image> readData(String path) {
        List<Image> images = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(path))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] parts = line.split(",");
                double[][] data = new double[rows][columns];
                int label = Integer.parseInt(parts[0]);
                int i = 1;
                for (int row = 0; row < rows; row++) {
                    for (int column = 0; column < columns; column++) {
                        data[row][column] = (double) Integer.parseInt(parts[i]);
                        i++;
                    }
                }
                images.add(new Image(data, label));
            }
            reader.close();
        } catch (Exception e) {
            System.out.println("Error reading file: " + e.getMessage());
        }

    return images;
    }

}
