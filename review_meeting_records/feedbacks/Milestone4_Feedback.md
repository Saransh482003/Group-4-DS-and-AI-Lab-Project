## Feedback for Milestone 2

- **Define which datasets will actually be used** – The report lists many datasets but does not clearly specify which ones will be used for training, validation, and testing.
- **Justify the use of very large datasets** – Extremely large datasets (e.g., Open Images, ScanNet, Common Voice) are listed without explaining whether subsets will be used or how they will be processed.
- **Describe dataset filtering strategy** – Explain how large datasets will be filtered to retain only the relevant indoor navigation classes.
- **Provide a unified label schema** – Multiple datasets use different label sets; the report must explain how these labels will be mapped to a consistent set of indoor navigation classes.
- **Explain annotation format standardization** – Different datasets use COCO JSON, YOLO format, XML, or RGB-D formats; the document should explain how these annotations will be converted into a common format.
- **Strengthen the custom dataset collection plan** – Provide details such as camera device, resolution, number of locations, annotation tools, and labeling guidelines.
- **Include dataset statistics and exploratory analysis** – Provide quantitative statistics such as number of images per class, bounding box counts, and objects per image.
- **Analyze class imbalance** – Show the distribution of object classes and describe how imbalance will be handled.
- **Explain processing of depth datasets** – Clarify how depth maps will be normalized, how missing depth values will be handled, and how depth units will be standardized.
- **Describe RGB–depth alignment procedures** – Explain how RGB images and depth maps will be aligned, resized, and calibrated across datasets.
- **Improve the train/validation/test split explanation** – Specify how splits will be performed across datasets and ensure frames from the same scene or sequence do not appear in multiple splits.
- **Provide a clear dataset integration strategy** – Explain how multiple datasets will be combined, whether they will be trained jointly or sequentially, and how domain differences will be handled.
- **Simplify and clarify the preprocessing pipeline** – The current section lists many complex restoration models; instead clearly define the practical preprocessing steps that will actually be applied.
- **Specify final input image resolution** – Since datasets have different resolutions, define the standard resolution used for model training.
- **Clarify the role of speech datasets** – Explain how large speech datasets will actually contribute to the navigation system and whether they are necessary for the task.
- **Verify dataset licensing and usage constraints** – Clearly confirm that all datasets are legally usable for research and compatible with the project.
- **Explain dataset storage and processing requirements** – Provide an estimate of dataset size and describe how the data will be stored and processed.
- **Define the evaluation dataset** – Clearly specify which dataset will be used to evaluate system performance in indoor navigation scenarios.
- **Add a dataset preparation pipeline diagram** – Include a visual pipeline showing dataset collection, filtering, preprocessing, annotation conversion, splitting, and training usage.

