using Microsoft.ML;
using Microsoft.ML.Data;

string _trainingFilePath = "C:\\Users\\Mallika_Panjwani\\source\\repos\\EmailSubjectClassifier\\EmailSubjectClassifier\\Models\\subjectModel.tsv";
string _modelFilePath = "C:\\Users\\Mallika_Panjwani\\source\\repos\\EmailSubjectClassifier\\EmailSubjectClassifier\\Models\\model.zip";

MLContext _mlContext;
IDataView _trainingDataView;
ITransformer _model;
PredictionEngine<EmailSubject, DepartmentPrediction> _predictionEngine;

_mlContext = new MLContext(seed:0);
_trainingDataView = _mlContext.Data.LoadFromTextFile<EmailSubject>(_trainingFilePath, hasHeader:true);
var pipeline = ProcessData();
var trainingPipeline = BuildAndTrainModel(_trainingDataView, pipeline);
SaveModelAsFile();
var result = PredictDepartmentForSubjectLine("New Invoice");

var keepRunning = true;
Console.WriteLine("Enter subject lines to predict. Type QUIT to close the app");
while (keepRunning)
{
    var subjectLine = Console.ReadLine();
    if(subjectLine == "QUIT")
    {
        keepRunning = false;
    }
    else
    {
        Console.WriteLine(PredictDepartmentForSubjectLine(subjectLine));
    }
}

string PredictDepartmentForSubjectLine(string subjectLine)
{
    var model = _mlContext.Model.Load(_modelFilePath, out var modelInputSchema);
    var emailSubject = new EmailSubject() { Subject = subjectLine };
    _predictionEngine = _mlContext.Model.CreatePredictionEngine<EmailSubject, DepartmentPrediction>(model);
    var result = _predictionEngine.Predict(emailSubject);
    return result.Department;
}
void SaveModelAsFile()
{
    _mlContext.Model.Save(_model,_trainingDataView.Schema, _modelFilePath);
}
IEstimator<ITransformer> ProcessData()
{
    var pipeline = _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Department", outputColumnName: "Label")
           .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Subject", outputColumnName: "EmailSubjectFeaturized"))
           .Append(_mlContext.Transforms.Concatenate("Features", "EmailSubjectFeaturized"))
           .AppendCacheCheckpoint(_mlContext);
    return pipeline;
}

IEstimator<ITransformer> BuildAndTrainModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline) {
    var trainingPipeline = pipeline.Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
        .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
    _model = trainingPipeline.Fit(trainingDataView);
    return pipeline;
}
public class EmailSubject
{
    [LoadColumn(0)]
    public string Subject { get; set; }
    [LoadColumn(1)]
    public string Department { get; set; }
}

public class DepartmentPrediction
{
    [ColumnName("PredictedLabel")]
    public string? Department { get; set; }
}