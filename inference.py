from options.test_options import TestOptions
from data.base_dataset import get_transform
from models import create_model
from models.cast_model import CASTModel
from util.util import copyconf, tensor2im
from PIL import Image

if __name__ == "__main__":
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.checkpoints_dir = "./CAST_model"
    opt.name = "cast"
    opt.preprocess = "none"

    model: CASTModel = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)
    model.eval()

    is_finetuning = False

    modified_opt = copyconf(opt, load_size=opt.crop_size if is_finetuning else opt.load_size)
    transform = get_transform(modified_opt)

    style_image_path = "style.jpg"
    content_image_path = "content.jpg"

    style_image = Image.open(style_image_path).convert("RGB")
    content_image = Image.open(content_image_path).convert("RGB")

    style_image = transform(style_image)[None, ...]
    content_image = transform(content_image)[None, ...]

    model.set_input({'A': content_image, 'B': style_image, 'A_paths': [], 'B_paths': []})  # unpack data from data loader
    model.test() 

    image = tensor2im(model.fake_B.cpu())
    Image.fromarray(image, mode="RGB").save("res.jpg")