?	\ A?c@\ A?c@!\ A?c@	p?He:@@p?He:@@!p?He:@@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$\ A?c@`vOj??A?ܵ?|?@Y?q?????*	?????Ε@2F
Iterator::Modelt??????!????R@)Gr?????1r?4?f?R@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?%䃞???!??n??5@)L?
F%u??1'Xr?ة4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat%u???!C^I?? @)?(??0??1o?6]n3??:Preprocessing2U
Iterator::Model::ParallelMapV2S?!?uq??!?M@???)S?!?uq??1?M@???:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??j+????!?K,??,8@)??0?*??1c.S???:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??_vOv?!ap????)??_vOv?1ap????:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora2U0*?s?!W?o????)a2U0*?s?1W?o????:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?D?????!鬉V?85@)a2U0*?c?1W?o????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 32.2% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*high2t15.8 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9p?He:@@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	`vOj??`vOj??!`vOj??      ??!       "      ??!       *      ??!       2	?ܵ?|?@?ܵ?|?@!?ܵ?|?@:      ??!       B      ??!       J	?q??????q?????!?q?????R      ??!       Z	?q??????q?????!?q?????JCPU_ONLYYp?He:@@b Y      Y@q????qL@"?	
host?Your program is HIGHLY input-bound because 32.2% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nohigh"t15.8 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.:
Refer to the TF2 Profiler FAQb?56.8887% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 