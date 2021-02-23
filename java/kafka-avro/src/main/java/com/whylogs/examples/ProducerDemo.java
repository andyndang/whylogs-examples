package com.whylogs.examples;

import com.whylabs;
import com.whylogs.core.DatasetProfile;
import org.apache.avro.Schema;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.kafka.clients.producer.ProducerConfig;

import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.time.LocalDate;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.*;

/**
 * An example of processing a CSV dataset.
 *
 * Here we demonstrate how you can extract data from a CSV file and track it with WhyLogs. We group
 * the data by year here and run profiling for each year.
 *
 * In practice, if the data is sorted by date, you can write the data to disk as soon as you see the timestamp
 * increase (in this case, you see the value of the following year in the dataset). In that way you can
 * guarantee constant memory usage.
 */
public class ProducerDemo {

    public static final String DATE_COLUMN = "Call Date";
    public static final CSVFormat CSV_FORMAT = CSVFormat.DEFAULT
            .withFirstRecordAsHeader()
            .withNullString("")
            .withDelimiter(',');
    public static final String INPUT_FILE_NAME = "Fire_Department_Calls_for_Service.csv";
    public static final DateTimeFormatter DATE_TIME_FORMAT = DateTimeFormatter.ofPattern("MM/dd/yyyy");

    public static void main(String[] args) throws Exception {
        final String sessionId = UUID.randomUUID().toString();
        final Instant now = Instant.now();


        Schema schema = value_lending_club.getClassSchema();

        Properties props = new Properties();
        props.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
        props.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG,
                org.apache.kafka.common.serialization.StringSerializer.class);
        props.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG,
                io.confluent.kafka.serializers.KafkaAvroSerializer.class);
        props.put("schema.registry.url", "http://localhost:8081");
        KafkaProducer producer = new KafkaProducer(props);

        // map for storing the result
        final Map<Instant, DatasetProfile> result = new HashMap<>();
        System.out.println("opening " + INPUT_FILE_NAME);

        try (final InputStreamReader is = new InputStreamReader(ProducerDemo.class.getResourceAsStream(INPUT_FILE_NAME))) {
            final CSVParser parser = new CSVParser(is, CSV_FORMAT);

            // iterate through records
            for (final CSVRecord record : parser) {
                // extract date time
                final Instant dataTime = parseAndTruncateToYear(record.get(DATE_COLUMN));
                
                // create new dataset profile
                final DatasetProfile profile = result.computeIfAbsent(dataTime,
                        t -> new DatasetProfile(sessionId, now, t, Collections.emptyMap(), Collections.emptyMap()));

                // track multiple features
                profile.track(record.toMap());
            }
        }

        System.out.println("Number of profiles: " + result.size());

        // write to a folder called "output"
        final Path output = Paths.get("output");
        Files.createDirectories(output);

        for (Map.Entry<Instant, DatasetProfile> entry : result.entrySet()) {
            final DatasetProfile profile = entry.getValue();
            // associate the year with filename
            final String fileName = String.format("profile_%s.bin", entry.getKey().atZone(ZoneOffset.UTC).getYear());

            // write out the output
            try (final OutputStream os =
                         Files.newOutputStream(output.resolve(fileName), StandardOpenOption.WRITE, StandardOpenOption.CREATE)) {
                profile.toProtobuf().build().writeDelimitedTo(os);
            }
        }
    }

    /**
     * Parse a text to an Instant object. This is used to extract data from the CSV and map
     * them into DatasetProfile's dataset_timestamp
     *
     * @param text input text
     * @return time in UTC as {@link Instant}
     */
    private static Instant parseAndTruncateToYear(String text) {
        return LocalDate.parse(text, DATE_TIME_FORMAT)
                .atStartOfDay()
                .withDayOfMonth(1)
                .withMonth(1)
                .atZone(ZoneOffset.UTC).toInstant();
    }
}
