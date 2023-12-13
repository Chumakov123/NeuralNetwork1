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
        private class Neuron
        {
            //Входной взвешенный сигнал
            public double charge = 0;
            //Выходной сигнал
            public double output = 0;
            //Ошибка
            public double error = 0;
            //Сигнал поляризации (нафига нужен и почему -1)
            public double biasSignal = 1.0;

            private static readonly Random rnd = new Random();
            //Диапазон инициализации весов (нужно разобраться почему -1 и 1)
            private const double initMinWeight = -1;
            private const double initMaxWeight = 1;
            //Количество узлов на предыдущем слое (можно избавиться)
            private readonly int inputLayerSize = 0;
            //Входные веса нейрона
            private readonly double[] weights = null;
            //Вес на сигнале поляризации (разобраться)
            private double biasWeight = 0.01;
            //Ссылка на предыдущий слой
            private readonly Neuron[] inputLayer = null;
            public Neuron(Neuron[] prevLayerNeurons)
            {
                inputLayer = prevLayerNeurons;
                if (prevLayerNeurons == null || prevLayerNeurons.Length == 0)
                    return;

                inputLayerSize = prevLayerNeurons.Length;
                weights = new double[inputLayerSize];

                for (int i = 0; i < weights.Length; ++i)
                {
                    weights[i] = initMinWeight + rnd.NextDouble() * (initMaxWeight - initMinWeight);
                }
            }
            //Вычисление пороговой функции (активация нейрона). Для сенсоров не используется
            public void Activate()
            {
                //Взвешенное значение сигнала поляризации
                charge = biasWeight * biasSignal;
                for (int i = 0; i < inputLayer.Length; ++i)
                {
                    charge += inputLayer[i].output * weights[i];
                }
                //Считаем выход нейрона
                output = ActivationFunc(charge);
                //Сброс входного сигнала
                charge = 0;
            }
            public void BackpropError(double ita)
            {
                //Сначала обрабатываем ошибку собственно в текущем нейроне
                error *= output * (1 - output);
                //Теперь разбираемся с сигналом поляризации - он имеет выход -1 и вес biasWeight, его пересчитыва
                biasWeight += ita * error * biasSignal;

                //Можно по формуле 2*alpha*out(1-out)*Сумма(ошибка в зависимых x вес)

                for (int i = 0; i < inputLayerSize; ++i)
                {
                    inputLayer[i].error += error * weights[i];
                    //Проброс ошибки на предыдущий слой 
                }
                for (int i = 0; i < inputLayerSize; ++i)
                {
                    weights[i] += ita * error * inputLayer[i].output;
                }
                error = 0;
            }
            //Функция активации (Сигмоида)
            private double ActivationFunc(double x)
            {
                return 1.0 / (1.0 + Math.Exp(-x));
            }
        }

        public double learningSpeed = 0.001; // Скорость обучения
        private Neuron[] sensors;
        private Neuron[] outputs;
        private Neuron[][] layers; //Здесь создаются нейроны, остальные массивы - ссылки

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
            sensors = layers[0];

            //Остальные слои, указывая каждому нейрону предыдущий слой
            for (int layer = 1; layer < structure.Length; ++layer)
            {
                layers[layer] = new Neuron[structure[layer]];
                for (int neuron = 0; neuron < structure[layer]; ++neuron)
                    layers[layer][neuron] = new Neuron(layers[layer - 1]);
            }
            //Ссылка на выходной слой
            outputs = layers[layers.Length - 1];
        }
        //Однократный запуск сети
        private double[] Run(Sample image)
        {
            double[] result = Compute(image.input);
            //Inversion(result);
            image.ProcessPrediction(result);
            return result;
        }

        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            int iterations = 0;

            Run(sample);
            double error = sample.EstimatedError();

            while (error > acceptableError)
            {
                Debug.WriteLine($"e {error} a {acceptableError}");
                Run(sample);
                error = sample.EstimatedError();

                ++iterations;
                BackProp(sample, learningSpeed);
            }
            return iterations;
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            Debug.WriteLine("Обучение");
            watch.Restart();
            double error = double.PositiveInfinity;

            for (int curEpoch = 0; curEpoch < epochsCount; ++curEpoch)
            {
                Debug.WriteLine($"Эпоха {curEpoch}");
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
        private void BackProp(Sample image, double ita)
        {
            // Считываем ошибку из образа на выходной слой
            for (int i = 0; i < outputs.Length; i++)
                outputs[i].error = image.error[i];

            // От выходов к корням
            for (int i = layers.Length - 1; i >= 0; --i)
                for (int j = 0; j < layers[i].Length; ++j)
                    layers[i][j].BackpropError(ita);
        }
        
        private double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
        private double SigmoidDerivative(double x) => x * (1 - x);

        private double GetError(double[] res, double[] expect) // Среднее квадратичное отклонение
        {
            double error = 0;
            for (int i = 0; i < res.Length; i++)
            {
                error += Math.Pow(expect[i] - res[i], 2);
            }
            return error / 2;
        }

    }
}