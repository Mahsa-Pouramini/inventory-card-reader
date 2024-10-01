from pero_ocr.user_scripts.parse_folder import get_device, PageParser, Computator, create_dir_if_not_exists
from configparser import ConfigParser
import os


class PeroOCRProcessor():
    def __init__(self, config_path, input_image_path, output_xml_path, device_name="gpu"):
        config = ConfigParser()
        config.read(config_path)

        self.device = get_device(device_name)

        page_parser = PageParser(config,config_path=os.path.dirname(config_path), device=self.device)
        create_dir_if_not_exists(output_xml_path)
        self.computator = Computator(page_parser, input_image_path=input_image_path, input_xml_path=None, input_logit_path=None, 
                                     output_render_path=None, output_logit_path=None, output_alto_path=None, output_xml_path=output_xml_path,
                                     output_line_path=None)

    def parse_directory(self, image_directory):
        image_exts = ['.jpg'] 
        images_to_process = [fn for fn in os.listdir(image_directory) if os.path.splitext(fn)[1] in image_exts]
        images_to_process = sorted(images_to_process)
        ids_to_process = [os.path.splitext(os.path.basename(fn))[0] for fn in images_to_process]

        results = []
        for index, (file_id, image_file_name) in enumerate(zip(ids_to_process, images_to_process)):
            results.append(self.computator(image_file_name, file_id, index, len(ids_to_process)))

        return results
        