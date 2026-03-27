from metaflow import FlowSpec, step, Parameter

class XRayPipeline(FlowSpec):

    train_dir = Parameter('train_dir', default='sample_data_img/train')
    val_dir   = Parameter('val_dir',   default='sample_data_img/val')
    img_size  = Parameter('img_size',  default=224)

    @step
    def start(self):
        print("STARTS .............")
        self.next(self.process_images)

    @step
    def process_images(self):
        print("PROCESS STEP - before import")
        from scripts.process_image import process_data
        self.data_conf = process_data(
            dir_train_path=self.train_dir,
            dir_validation_path=self.val_dir,
            img_size=(self.img_size, self.img_size)
        )
        self.next(self.end)

    @step
    def end(self):
        print("Done")
        print(self.data_conf)

if __name__ == '__main__':
    XRayPipeline()
