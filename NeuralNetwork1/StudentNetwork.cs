using Accord.Neuro;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;

namespace NeuralNetwork1
{
    public class StudentNetwork : BaseNetwork
    {
        private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
        private static double SigmoidDerivative(double x) => x * (1 - x);
        public static double learningSpeed = 0.01;
        private static Random r = new Random();
        private class Neuron
        {
            public Neuron[] inputs;
            public double[] weights;
            public double error;
            public double biasWeight;
            public double output;
            public void Activate()
            {
                    double weightedSum = 0;

                    for (int i = 0; i < inputs.Length; ++i)
                    {
                        weightedSum += inputs[i].output * weights[i];
                    }
                    weightedSum += biasWeight;

                    output = Sigmoid(weightedSum);
            }
            public Neuron(Neuron[] prevLayerNeurons)
            {
                if (prevLayerNeurons == null || prevLayerNeurons.Length == 0)
                    return;
                inputs = prevLayerNeurons;
                weights = new double[inputs.Length];
                randomizeWeights();
            }
            private void randomizeWeights()
            {
                for (int i = 0; i < weights.Length; ++i)
                {
                    weights[i] = 0.2 * r.NextDouble() - 0.1;
                }
                biasWeight = 0.2 * r.NextDouble() - 0.1;
            }
            public void AdjustWeights()
            {
                for (int i = 0; i < weights.Length; ++i)
                {
                    weights[i] -= learningSpeed * error * inputs[i].output;
                }
                biasWeight -= learningSpeed * error;
            }
        }

        private Neuron[] sensors;
        private Neuron[] outputs;
        private Neuron[][] layers; //Здесь создаются нейроны, остальные массивы - просто ссылки

        private readonly Stopwatch watch = new Stopwatch();

        public StudentNetwork(int[] structure)
        {
            InitializeNetwork(structure);
        }
        public StudentNetwork(string path)
        {
            //TODO LoadFromFile(path);
        }
        private void InitializeNetwork(int[] structure)
        {
            if (structure.Length < 2)
                throw new Exception("Invalid initialize structure");

            layers = new Neuron[structure.Length][];

            //Сенсоры
            layers[0] = new Neuron[structure[0]];
            for (int neuron = 0; neuron < structure[0]; ++neuron)
                layers[0][neuron] = new Neuron(null);

            //Остальные слои, указывая каждому нейрону предыдущий слой
            for (int layer = 1; layer < structure.Length; ++layer)
            {
                layers[layer] = new Neuron[structure[layer]];
                for (int neuron = 0; neuron < structure[layer]; ++neuron)
                    layers[layer][neuron] = new Neuron(layers[layer - 1]);
            }
            //Ссылки на входной и выходной слои
            sensors = layers[0];
            outputs = layers[layers.Length - 1];
        }
        //Однократный запуск сети
        private double[] Run(Sample image)
        {
            double[] result = Compute(image.input);
            image.ProcessPrediction(result);
            return result;
        }
        //Обучение на одном элементе
        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            int iterations = 0;

            Run(sample);
            double error = sample.EstimatedError();

            while (error > acceptableError)
            {
                //Debug.WriteLine($"e {error} a {acceptableError}");
                Run(sample);
                error = sample.EstimatedError();

                ++iterations;
                BackProp(sample);
            }
            return iterations;
        }
        //Обучение на наборе
        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            //Debug.WriteLine("Обучение");
            watch.Restart();
            double error = double.PositiveInfinity;

            for (int curEpoch = 0; curEpoch < epochsCount; ++curEpoch)
            {
                //Debug.WriteLine($"Эпоха {curEpoch}");
                double errorSum = 0;
                for (int i = 0; i < samplesSet.Count; ++i)
                {
                    if (Train(samplesSet.samples.ElementAt(i), acceptableError, false) == 0)
                        errorSum += samplesSet.samples.ElementAt(i).EstimatedError();
                }
                error = errorSum;
                OnTrainProgress(((curEpoch+1) * 1.0) / epochsCount, error, watch.Elapsed);
            }
            watch.Stop();
            return error;
        }

        //Ответ нейросети на входные данные
        protected override double[] Compute(double[] input)
        {
            //Передаем значениея сенсорам
            for (int i = 0; i < input.Length; ++i)
                sensors[i].output = input[i];

            //Обрабатываем все остальные слои
            for (int i = 1; i < layers.Length; ++i)
                for (int j = 0; j < layers[i].Length; ++j)
                    layers[i][j].Activate();

            return outputs.Select(x => x.output).ToArray();
        }
        //Обратное распространение ошибки
        private void BackProp(Sample image)
        {
            //Ошибка выходного слоя
            for (int i = 0; i < outputs.Length; ++i)
            {
                outputs[i].error = image.error[i];
            }

            // Рассчитываем ошибки для скрытых слоев
            for (int layer = layers.Length - 2; layer > 0; --layer)
            {
                for (int j = 0; j < layers[layer].Length; ++j)
                {
                    double sum = 0;
                    for (int k = 0; k < layers[layer + 1].Length; ++k)
                    {
                        sum += layers[layer + 1][k].error * layers[layer + 1][k].weights[j];
                    }
                    layers[layer][j].error = sum * SigmoidDerivative(layers[layer][j].output);
                }
            }

            // Обновляем веса для выходного слоя
            for (int i = 0; i < outputs.Length; ++i)
            {
                outputs[i].AdjustWeights();
            }

            // Обновляем веса для скрытых слоев
            for (int layer = layers.Length - 2; layer > 0; --layer)
            {
                for (int j = 0; j < layers[layer].Length; ++j)
                {
                    layers[layer][j].AdjustWeights();
                }
            }
        }
    }
}