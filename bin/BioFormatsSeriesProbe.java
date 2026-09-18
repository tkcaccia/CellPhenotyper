import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import loci.formats.ImageReader;
import loci.formats.MetadataTools;
import loci.formats.meta.IMetadata;
import ome.units.quantity.Length;
import org.slf4j.LoggerFactory;

/** Print one unflattened Bio-Formats series as a single JSON object. */
public final class BioFormatsSeriesProbe {
    private static String numberOrNull(Length value) {
        if (value == null || value.value() == null) return "null";
        return value.value().toString();
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 2) throw new IllegalArgumentException("PATH SERIES");
        Logger root = (Logger) LoggerFactory.getLogger(Logger.ROOT_LOGGER_NAME);
        root.setLevel(Level.ERROR);
        String path = args[0];
        int series = Integer.parseInt(args[1]);
        ImageReader reader = new ImageReader();
        // bioformats2raw selects logical images with sub-resolutions attached;
        // the default flattened view would incorrectly count each overview as
        // an independent series.
        reader.setFlattenedResolutions(false);
        IMetadata metadata = MetadataTools.createOMEXMLMetadata();
        reader.setMetadataStore(metadata);
        try {
            reader.setId(path);
            if (series < 0 || series >= reader.getSeriesCount()) {
                throw new IllegalArgumentException("series out of range: " + series);
            }
            reader.setSeries(series);
            System.out.printf(
                "{\"series_index\":%d,\"series_count\":%d,\"width_px\":%d,\"height_px\":%d," +
                "\"size_c\":%d,\"size_z\":%d,\"size_t\":%d," +
                "\"physical_size_x_um\":%s,\"physical_size_y_um\":%s," +
                "\"backend\":\"bioformats_unflattened_series\"}%n",
                series, reader.getSeriesCount(), reader.getSizeX(), reader.getSizeY(),
                reader.getSizeC(), reader.getSizeZ(), reader.getSizeT(),
                numberOrNull(metadata.getPixelsPhysicalSizeX(series)),
                numberOrNull(metadata.getPixelsPhysicalSizeY(series))
            );
        } finally {
            reader.close();
        }
    }
}
