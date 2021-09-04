class TestMnistTrain : public ::testing::Test {
protected:
    std::shared_ptr<cg::ComputingGraph> graph;

    size_t batch_size = 1;
    opr::AddUpdate::SharedScalar lr = std::make_shared<DTypeScalar>(.0f);

    std::shared_ptr<HostTensorND> m_input, m_label, m_cost, m_conv1, m_conv2,
            m_fc1, m_fc2, m_fc3, m_loss, m_bias1, m_bias2, m_bias3;
    std::vector<std::shared_ptr<HostTensorND>> m_grads;

    HostTensorGenerator<dtype::Float32, RandomDistribution::GAUSSIAN>
            gen_float32;
    HostTensorGenerator<dtype::Int32> gen_int32;
    TensorGeneratorExpansion<dtype::Float32> gen_ex_float32;
    TensorGeneratorExpansion<dtype::Int32> gen_ex_int32;

    TensorShape conv1_shape = {6, 1, 5, 5};
    TensorShape conv2_shape = {16, 6, 5, 5};
    TensorShape fc1_shape = {1, 120, 16 * 4 * 4};
    TensorShape fc2_shape = {1, 32, 120};
    TensorShape fc3_shape = {1, 10, 32};
    TensorShape input_shape = {1, 1, 28, 28};
    TensorShape label_shape = {1, 1};
    TensorShape fc1_bias_shape = {1, 120, 1};
    TensorShape fc2_bias_shape = {1, 32, 1};
    TensorShape fc3_bias_shape = {1, 10, 1};

    cg::ComputingGraph::OutputSpec spec_train;
    cg::ComputingGraph::OutputSpec spec_cost;
    cg::ComputingGraph::OutputSpec spec_result;
    cg::ComputingGraph::OutputSpec spec;

    std::unique_ptr<cg::AsyncExecutable> func;

    std::shared_ptr<HostTensorND> m_host;

    float err = .0f;
    int pred = -1;

    std::vector<CompNode> cns = load_multiple_xpus(3);

    enum Mode : uint32_t { TRAIN = 0, COST = 1, RESULT = 2 };

    void build_model();

    std::unique_ptr<cg::AsyncExecutable> set_mode(Mode mode);

    void train(int epochs);

    void evaluate();
};
}  // namespace

void TestMnistTrain::evaluate() {
    if (graph->nr_oprs_in_graph() == 0) {
        build_model();
    }
    std::cout << "Start evaluate" << std::endl;
    auto test_dataset =
            mnist_dataset(mnist_dataset::Mode::TEST, dataset_dir, 1, false);

    int correct_num = 0;
    //     auto func = set_mode(Mode::RESULT);
    func = graph->compile(spec_result);
    std::cout << 1111 << std::endl;
    for (auto data : test_dataset.get()) {
        // read input data
        m_input = data.first;
        m_label = data.second;

        // run the graph
        func->execute();
        if (std::abs(pred - m_label->ptr<float>()[0]) < 0.05f) {
            correct_num += 1;
            std::cout << "pred: " << pred
                      << " label: " << m_label->ptr<float>()[0] << std::endl;
        }
    }
    std::cout << "Accuracy rate: "
              << (double)correct_num / test_dataset.get().size() << std::endl;
}

void TestMnistTrain::train(int epochs) {
    std::cout << 51 << std::endl;
    lr->set<float>(1.0f);
    std::cout << 52 << std::endl;
    // construct the computing graph
    build_model();
    std::cout << "The weight: " << std::endl;
    auto ptr = m_fc3->ptr<float>();
    for (size_t i = 0, it = fc3_shape.total_nr_elems(); i < it; i += 1) {
        std::cout << ptr[i] << std::endl;
    }

    // declare the dataset
    auto train_dataset =
            mnist_dataset(mnist_dataset::Mode::TRAIN, dataset_dir, 1, true);

    std::cout << "start training." << std::endl;

    // TODO: lr scheduler

    func = graph->compile(spec);

    func->to_json()->writeto_fpath(output_file(ssprintf("train-mnist.json")));

    for (int epoch = 0; epoch < epochs; epoch++) {
        int time = 0;
        for (auto data : train_dataset.get()) {
            // read input data
            time++;
            m_input->copy_from(*(data.first.get())).sync();
            m_label->copy_from(*(data.second.get())).sync();

            // run the graph
            func->execute();
        }
        std::cout << "epoch: " << epoch << " complete." << std::endl;
    }
    std::cout << "The weight: " << std::endl;
    ptr = m_fc3->ptr<float>();
    for (size_t i = 0, it = fc3_shape.total_nr_elems(); i < it; i += 1) {
        std::cout << ptr[i] << std::endl;
    }
}



void TestMnistTrain::build_model() {
    graph = cg::ComputingGraph::make();

    // initialize weights and necessary variable in the graph
    m_conv1 = gen_float32(conv1_shape);
    m_conv2 = gen_float32(conv2_shape);
    m_fc1 = gen_float32(fc1_shape);
    m_fc2 = gen_float32(fc2_shape);
    m_fc3 = gen_float32(fc3_shape);
    m_input = gen_float32(input_shape);
    m_label = gen_ex_float32.zeros(CompNode::load("xpu0"), label_shape);
    m_host = std::make_shared<HostTensorND>(CompNode::load("xpu0"));
    m_loss = gen_ex_float32.zeros(CompNode::load("xpu0"), {1});
    m_bias1 = gen_float32(fc1_bias_shape);
    m_bias2 = gen_float32(fc2_bias_shape);
    m_bias3 = gen_float32(fc3_bias_shape);


    // set params
    opr::Pooling::Param pooling_param(megdnn::Pooling::Param::Mode::MAX, 0, 0,
                                      2, 2, 2, 2);
    opr::Pooling::ExecutionPolicy::Strategy pooling_strategy =
            opr::Pooling::ExecutionPolicy::Strategy::PROFILE;
    opr::Pooling::ExecutionPolicy pooling_policy;
    pooling_policy.strategy = pooling_strategy;

    // define symbolvars
    // first conv layer
    SymbolVar symbol_input = opr::Host2DeviceCopy::make(*graph, m_input);
    SymbolVar symbol_label = opr::Host2DeviceCopy::make(*graph, m_label);
    SymbolVar symbol_loss = opr::SharedDeviceTensor::make(*graph, *m_loss);
    SymbolVar symbol_filter1 = opr::SharedDeviceTensor::make(*graph, *m_conv1);
    SymbolVar symbol_conv1 =
            opr::Convolution::make(symbol_input, symbol_filter1);
    SymbolVar symbol_relu1 = opr::relu(symbol_conv1);
    SymbolVar symbol_maxpool1 =
            opr::Pooling::make(symbol_relu1, pooling_param, {}, pooling_policy);

    // second conv layer
    SymbolVar symbol_filter2 = opr::SharedDeviceTensor::make(*graph, *m_conv2);
    SymbolVar symbol_conv2 =
            opr::Convolution::make(symbol_maxpool1, symbol_filter2);
    SymbolVar symbol_relu2 = opr::relu(symbol_conv2);
    SymbolVar symbol_maxpool2 =
            opr::Pooling::make(symbol_relu2, pooling_param, {}, pooling_policy);

    // flatten layer
    SymbolVar symbol_flatten = opr::Reshape::make(symbol_maxpool2, {1, 256, 1});

    // first fc layer
    SymbolVar symbol_weight_fc1 = opr::SharedDeviceTensor::make(*graph, *m_fc1);
    SymbolVar symbol_weight_bias1 = opr::SharedDeviceTensor::make(*graph, *m_bias1);
    SymbolVar symbol_fc1 =
            opr::add(opr::BatchedMatrixMul::make(symbol_weight_fc1, symbol_flatten), symbol_weight_bias1);
    SymbolVar symbol_relu3 = opr::relu(symbol_fc1);

    // second fc layer
    SymbolVar symbol_weight_fc2 = opr::SharedDeviceTensor::make(*graph, *m_fc2);
    SymbolVar symbol_weight_bias2 = opr::SharedDeviceTensor::make(*graph, *m_bias2);
    SymbolVar symbol_fc2 =
            opr::add(opr::BatchedMatrixMul::make(symbol_weight_fc2, symbol_relu3), symbol_weight_bias2);
    SymbolVar symbol_relu4 = opr::relu(symbol_fc2);

    // third fc layer(output layer)
    SymbolVar symbol_weight_fc3 = opr::SharedDeviceTensor::make(*graph, *m_fc3);
    SymbolVar symbol_weight_bias3 = opr::SharedDeviceTensor::make(*graph, *m_bias3);
    SymbolVar symbol_fc3 =
            opr::add(opr::BatchedMatrixMul::make(symbol_weight_fc3, symbol_relu4), symbol_weight_bias3);

    opr::Argmax::Param param;
    param.axis = 1;
    SymbolVar symbol_temp = opr::Copy::make(symbol_fc3);
    SymbolVar symbol_output = opr::Argmax::make(symbol_fc3, param);


    // define the callback
    auto ErrCallback = [&](DeviceTensorND& data) {
        std::shared_ptr<HostTensorND> m_host;
        m_host->copy_from(data).sync();
        err = m_host->ptr<float>()[0];
    };
    auto PredictionCallback = [&](DeviceTensorND& data) {
        m_host->copy_from(data).sync();
        pred = m_host->ptr<int>()[0];
    };
    auto EmptyCallback = [](DeviceTensorND& data) {
        std::cout << "The grad: ";
        auto ptr = data.ptr<float>();
        for (size_t i = 0, it = data.shape().total_nr_elems(); i < it && i < 60;
             i += 1) {
            std::cout << ptr[i] << "  ";
        }
        std::cout << data.layout().to_string();
        std::cout << std::endl;
    };
    auto EmptyCallback2 = [](DeviceTensorND& data) {
        std::cout << "The grad: ";
        auto ptr = data.ptr<int>();
        for (size_t i = 0, it = data.shape().total_nr_elems(); i < it && i < 60;
             i += 1) {
            std::cout << ptr[i] << "  ";
        }
        std::cout << std::endl;
    };



    spec.push_back({symbol_temp, EmptyCallback});
    spec.push_back({symbol_output, EmptyCallback2});

    spec_result.push_back({symbol_output, PredictionCallback});
}

TEST_F(TestMnistTrain, MnistLenetTrain) {
    train(1);
    evaluate();
}
