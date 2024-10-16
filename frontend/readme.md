# Single Page Vue App

## Components

### Upload
- Upload PPTX and PDF files
- Choose number of pages
- Select a model from a list
- Click 'Next' to navigate to the Doc component and send data to the backend

### Doc
- Display two text frames (left and right) with data from the backend
- Allow user to edit the data and send updates to the backend
- Click 'Next' to navigate to the Generate component

### Generate
- Show a progress bar indicating the generation process
- Receive the generated PPTX file from the backend and provide it for download