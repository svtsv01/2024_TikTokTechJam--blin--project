import express from "express";
import { exec } from 'child_process';

const app = express();
const port = 3000;

app.use(express.static("public"));
app.use(express.urlencoded({ extended: true }));

app.get("/", (req, res) => {
  res.render("index.ejs", { number:  null});
});

app.post('/extract-text', (req, res) => {
  const link = req.body.link;
  console.log('Link submitted:', link);
  //res.render("index.ejs", { number:  2});
  // Run the Python script with the link as an argument
  exec(`python script.py "${link}"`, (error, stdout, stderr) => {
      console.log(`Python script output: ${stdout}`);
      if(!stdout){
        res.render("index.ejs", { number:  -1});
      } else{
      res.render("index.ejs", { number:  `${stdout}`});
      }
  })
  
});

app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
