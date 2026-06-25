import json
import os
from config import EVAL_DATASET_PATH, DATA_DIR

# Define a comprehensive, high-quality evaluation dataset of 105 QA pairs.
# These are hand-crafted to align exactly with the downloaded corpus of 119 CS/ML articles.
EVAL_QA_PAIRS = [
    # === DIRECT QA (35 pairs) ===
    {
        "id": "direct_1",
        "category": "direct",
        "question": "Who is known as the father of information theory?",
        "ground_truth": "Claude Shannon is widely known as the father of information theory, having published the seminal paper 'A Mathematical Theory of Communication' in 1948.",
        "expected_sources": ["Claude_Shannon.txt", "Information_theory.txt"]
    },
    {
        "id": "direct_2",
        "category": "direct",
        "question": "What does Big O notation describe in computer science?",
        "ground_truth": "Big O notation describes the limiting behavior of a function when the argument tends towards a particular value or infinity. In computer science, it is used to classify algorithms according to how their run time or space requirements grow as the input size grows.",
        "expected_sources": ["Big_O_notation.txt", "Algorithm.txt"]
    },
    {
        "id": "direct_3",
        "category": "direct",
        "question": "What is the main difference between symmetric and asymmetric cryptography?",
        "ground_truth": "Symmetric cryptography uses the same secret key for both encryption and decryption, whereas asymmetric cryptography (public-key cryptography) uses a pair of keys: a public key for encryption and a private key for decryption.",
        "expected_sources": ["Cryptography.txt", "Public-key_cryptography.txt", "Symmetric-key_algorithm.txt"]
    },
    {
        "id": "direct_4",
        "category": "direct",
        "question": "Who invented the Turing machine and in what year?",
        "ground_truth": "Alan Turing invented the Turing machine in 1936.",
        "expected_sources": ["Alan_Turing.txt", "Turing_machine.txt"]
    },
    {
        "id": "direct_5",
        "category": "direct",
        "question": "What is the core architecture of a Transformer model?",
        "ground_truth": "The core architecture of a Transformer model is based on self-attention mechanisms, dispensing with recurrent or convolutional structures, consisting of an encoder and a decoder stack.",
        "expected_sources": ["Transformer_deep_learning_architecture.txt", "Attention_machine_learning.txt"]
    },
    {
        "id": "direct_6",
        "category": "direct",
        "question": "What is the difference between supervised and unsupervised learning?",
        "ground_truth": "Supervised learning uses labeled training data where each input is paired with its correct output. Unsupervised learning models analyze unlabeled data to discover hidden patterns or intrinsic structures.",
        "expected_sources": ["Supervised_learning.txt", "Unsupervised_learning.txt", "Machine_learning.txt"]
    },
    {
        "id": "direct_7",
        "category": "direct",
        "question": "What does the SQL acronym stand for?",
        "ground_truth": "SQL stands for Structured Query Language.",
        "expected_sources": ["SQL.txt", "Database.txt"]
    },
    {
        "id": "direct_8",
        "category": "direct",
        "question": "What is the primary purpose of a Docker container?",
        "ground_truth": "A Docker container packages an application and its dependencies together in a virtualized, isolated environment that can run consistently on any Linux or Windows system.",
        "expected_sources": ["Docker_software.txt", "Virtual_machine.txt"]
    },
    {
        "id": "direct_9",
        "category": "direct",
        "question": "What is the halting problem in computer science?",
        "ground_truth": "The halting problem is the problem of determining, from a description of an arbitrary computer program and an input, whether the program will finish running or continue to run forever. Alan Turing proved in 1936 that a general algorithm to solve the halting problem for all possible program-input pairs cannot exist.",
        "expected_sources": ["Halting_problem.txt", "Turing_machine.txt"]
    },
    {
        "id": "direct_10",
        "category": "direct",
        "question": "What is the purpose of the RSA cryptosystem?",
        "ground_truth": "RSA (Rivest-Shamir-Adleman) is an asymmetric public-key cryptosystem used for secure data transmission and digital signatures. It relies on the mathematical difficulty of factoring the product of two large prime numbers.",
        "expected_sources": ["RSA_cryptosystem.txt", "Public-key_cryptography.txt"]
    },
    {
        "id": "direct_11",
        "category": "direct",
        "question": "What is the purpose of TF-IDF in information retrieval?",
        "ground_truth": "TF-IDF (term frequency-inverse document frequency) is a numerical statistic intended to reflect how important a word is to a document in a collection or corpus, by balancing the local word frequency against its global document frequency.",
        "expected_sources": ["Tf-idf.txt", "Information_retrieval.txt"]
    },
    {
        "id": "direct_12",
        "category": "direct",
        "question": "Explain K-means clustering.",
        "ground_truth": "K-means clustering is a method of vector quantization and unsupervised learning that aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean (centroid).",
        "expected_sources": ["K-means_clustering.txt", "Unsupervised_learning.txt"]
    },
    {
        "id": "direct_13",
        "category": "direct",
        "question": "What is overfitting in machine learning?",
        "ground_truth": "Overfitting is the production of an analysis or machine learning model that corresponds too closely or exactly to a particular set of data, and may therefore fail to fit additional data or predict future observations reliably.",
        "expected_sources": ["Overfitting.txt", "Regularization_mathematics.txt"]
    },
    {
        "id": "direct_14",
        "category": "direct",
        "question": "What is the difference between TCP and UDP?",
        "ground_truth": "TCP (Transmission Control Protocol) is connection-oriented, reliable, and guarantees ordered delivery of packets using error checking and flow control. UDP (User Datagram Protocol) is connectionless, lightweight, and does not guarantee delivery or packet order, making it faster but less reliable.",
        "expected_sources": ["Transmission_Control_Protocol.txt", "User_Datagram_Protocol.txt", "Internet_protocol_suite.txt"]
    },
    {
        "id": "direct_15",
        "category": "direct",
        "question": "What is the purpose of Git?",
        "ground_truth": "Git is a distributed version control system designed to track changes in source code during software development, coordinate work among programmers, and maintain a history of code revisions.",
        "expected_sources": ["Git.txt", "Software_engineering.txt"]
    },
    {
        "id": "direct_16",
        "category": "direct",
        "question": "What is the Singleton pattern in software design?",
        "ground_truth": "The Singleton pattern is a software creational design pattern that restricts the instantiation of a class to one 'single' instance and provides a global point of access to that instance.",
        "expected_sources": ["Singleton_pattern.txt", "Software_design_pattern.txt"]
    },
    {
        "id": "direct_17",
        "category": "direct",
        "question": "What is Redis primarily used for?",
        "ground_truth": "Redis is an open-source, in-memory data structure store used as a database, cache, message broker, and queue, known for its extremely low latency.",
        "expected_sources": ["Redis.txt", "Caching.txt"]
    },
    {
        "id": "direct_18",
        "category": "direct",
        "question": "What is the P versus NP problem?",
        "ground_truth": "The P versus NP problem is a major unsolved problem in theoretical computer science. It asks whether every problem whose solution can be quickly verified by a computer (NP) can also be quickly solved by a computer (P).",
        "expected_sources": ["P_versus_NP_problem.txt", "Complexity_theory.txt"]
    },
    {
        "id": "direct_19",
        "category": "direct",
        "question": "What is a vector database?",
        "ground_truth": "A vector database is a specialized database designed to store, manage, and query high-dimensional vector embeddings, allowing for efficient similarity searches based on semantic distance.",
        "expected_sources": ["Vector_database.txt", "Semantic_search.txt"]
    },
    {
        "id": "direct_20",
        "category": "direct",
        "question": "What does ETL stand for?",
        "ground_truth": "ETL stands for Extract, Transform, Load, which is a three-step data integration process used to blend data from multiple sources into a single, consistent data store like a data warehouse.",
        "expected_sources": ["Extract,_transform,_load.txt", "Data_warehouse.txt", "Data_pipeline.txt"]
    },
    {
        "id": "direct_21",
        "category": "direct",
        "question": "What is a smart contract?",
        "ground_truth": "A smart contract is a computer program or a transaction protocol that is intended to automatically execute, control or document legally relevant events and actions according to the terms of a contract or agreement, running on a decentralized blockchain.",
        "expected_sources": ["Smart_contract.txt", "Blockchain.txt", "Ethereum.txt"]
    },
    {
        "id": "direct_22",
        "category": "direct",
        "question": "What is functional programming?",
        "ground_truth": "Functional programming is a programming paradigm where programs are constructed by applying and composing functions, emphasizing pure functions, avoiding shared state, mutable data, and side effects.",
        "expected_sources": ["Functional_programming.txt", "Object-oriented_programming.txt"]
    },
    {
        "id": "direct_23",
        "category": "direct",
        "question": "Who created the Git version control system?",
        "ground_truth": "Linus Torvalds created Git in 2005.",
        "expected_sources": ["Git.txt", "Linux.txt"]
    },
    {
        "id": "direct_24",
        "category": "direct",
        "question": "What is a major advantage of NoSQL databases?",
        "ground_truth": "NoSQL databases provide a mechanism for storage and retrieval of data that is modeled in means other than the tabular relations used in relational databases. They offer advantages like horizontal scalability, flexible schemas, and high write performance.",
        "expected_sources": ["NoSQL.txt", "Database.txt"]
    },
    {
        "id": "direct_25",
        "category": "direct",
        "question": "What is the Chomsky hierarchy?",
        "ground_truth": "The Chomsky hierarchy is a containment hierarchy of classes of formal grammars in theoretical computer science and linguistics, categorizing grammars into regular, context-free, context-sensitive, and recursively enumerable types.",
        "expected_sources": ["Chomsky_hierarchy.txt", "Formal_language.txt", "Automata_theory.txt"]
    },
    {
        "id": "direct_26",
        "category": "direct",
        "question": "What is the primary role of a garbage collector in a programming language runtime?",
        "ground_truth": "The primary role of a garbage collector is to automatically manage memory by reclaiming memory allocated to objects that are no longer in use by the program, preventing memory leaks.",
        "expected_sources": ["Garbage_collection_computer_science.txt", "Virtual_machine.txt"]
    },
    {
        "id": "direct_27",
        "category": "direct",
        "question": "Explain the concept of cosine similarity.",
        "ground_truth": "Cosine similarity measures the similarity between two non-zero vectors of an inner product space. It is calculated as the cosine of the angle between them, which determines whether two vectors are pointing in roughly the same direction.",
        "expected_sources": ["Cosine_similarity.txt", "Vector_database.txt", "Semantic_search.txt"]
    },
    {
        "id": "direct_28",
        "category": "direct",
        "question": "What is the purpose of regularizing a machine learning model?",
        "ground_truth": "Regularization is the process of adding information or a penalty term to a loss function to prevent overfitting by discouraging extreme parameter weights, simplifying the model.",
        "expected_sources": ["Regularization_mathematics.txt", "Overfitting.txt"]
    },
    {
        "id": "direct_29",
        "category": "direct",
        "question": "What is a decision tree model?",
        "ground_truth": "A decision tree is a non-parametric supervised learning method used for classification and regression. It models decisions and their possible consequences as a tree structure of branch nodes and leaves.",
        "expected_sources": ["Decision_tree.txt", "Random_forest.txt"]
    },
    {
        "id": "direct_30",
        "category": "direct",
        "question": "What is the purpose of the Domain Name System (DNS)?",
        "ground_truth": "DNS translates human-readable domain names (like www.wikipedia.org) into numerical IP addresses (like 198.35.26.96) that computers use to identify each other on the network.",
        "expected_sources": ["Domain_Name_System.txt", "Computer_network.txt"]
    },
    {
        "id": "direct_31",
        "category": "direct",
        "question": "What does a load balancer do?",
        "ground_truth": "A load balancer distributes network or application traffic across multiple servers to improve responsiveness, resource utilization, and reliability of applications.",
        "expected_sources": ["Load_balancing_computing.txt", "Computer_network.txt"]
    },
    {
        "id": "direct_32",
        "category": "direct",
        "question": "What is public-key cryptography?",
        "ground_truth": "Public-key cryptography, or asymmetric cryptography, is a cryptographic system that uses pairs of keys: public keys which may be disseminated widely, and private keys which are known only to the owner.",
        "expected_sources": ["Public-key_cryptography.txt", "Cryptography.txt"]
    },
    {
        "id": "direct_33",
        "category": "direct",
        "question": "Explain the concept of reinforcement learning.",
        "ground_truth": "Reinforcement learning is an area of machine learning concerned with how intelligent agents ought to take actions in an environment to maximize the notion of cumulative reward.",
        "expected_sources": ["Reinforcement_learning.txt", "Markov_decision_process.txt", "Q-learning.txt"]
    },
    {
        "id": "direct_34",
        "category": "direct",
        "question": "What is technical debt in software engineering?",
        "ground_truth": "Technical debt is a concept in software development that reflects the implied cost of additional rework caused by choosing an easy (quick) solution now instead of using a better approach that would take longer.",
        "expected_sources": ["Technical_debt.txt", "Software_engineering.txt"]
    },
    {
        "id": "direct_35",
        "category": "direct",
        "question": "What does a compiler do?",
        "ground_truth": "A compiler translates computer code written in one programming language (source language, e.g., C++) into another language (target language, e.g., machine code) usually to create an executable program.",
        "expected_sources": ["Compiler.txt", "Interpreter_computing.txt"]
    },

    # === MULTI-HOP QA (25 pairs) ===
    {
        "id": "multihop_1",
        "category": "multi-hop",
        "question": "How did Alan Turing's theoretical machine impact the concept of computability, and how does the halting problem define its limits?",
        "ground_truth": "Alan Turing introduced the Turing machine in 1936 to model formal computation, establishing that any algorithm can be executed by such a machine (Turing completeness). He then defined the limits of this model by proving that the halting problem—determining if an arbitrary program halts—is undecidable, meaning there are mathematically well-defined problems that cannot be solved by any computer.",
        "expected_sources": ["Alan_Turing.txt", "Turing_machine.txt", "Halting_problem.txt", "Computability_theory.txt"]
    },
    {
        "id": "multihop_2",
        "category": "multi-hop",
        "question": "How do Docker and Kubernetes collaborate in a microservices architecture?",
        "ground_truth": "Docker is used to package microservices and their dependencies into self-contained container images that run consistently. Kubernetes is a container orchestration platform that coordinates these Docker containers across clusters, managing scaling, networking, and high availability.",
        "expected_sources": ["Docker_software.txt", "Kubernetes.txt", "Microservices.txt"]
    },
    {
        "id": "multihop_3",
        "category": "multi-hop",
        "question": "Compare the data model and consistency strategies between SQL relational databases and NoSQL database types.",
        "ground_truth": "SQL relational databases use structured tabular schemas and support ACID transactions to ensure strict consistency. NoSQL databases (like document or graph stores) use flexible, schema-less data structures and often sacrifice strict consistency for high horizontal scalability, adopting eventual consistency.",
        "expected_sources": ["SQL.txt", "Database.txt", "Relational_database.txt", "NoSQL.txt"]
    },
    {
        "id": "multihop_4",
        "category": "multi-hop",
        "question": "How does self-attention in a Transformer solve the limitations of recurrent neural networks for NLP?",
        "ground_truth": "Recurrent neural networks (RNNs) process text sequentially, which restricts parallelization and leads to vanishing gradients over long sequences. The Transformer's self-attention mechanism processes all words in parallel, calculating relationships between all words directly and capturing long-range dependencies far more effectively.",
        "expected_sources": ["Transformer_deep_learning_architecture.txt", "Attention_machine_learning.txt", "Natural_language_processing.txt", "Recurrent_neural_network.txt"]
    },
    {
        "id": "multihop_5",
        "category": "multi-hop",
        "question": "Explain the role of public-key cryptography in establishing a TLS connection.",
        "ground_truth": "During a TLS handshake, public-key cryptography (like RSA or Diffie-Hellman) is used to securely authenticate the server's identity and negotiate a shared symmetric key, which is then used for fast encryption of the session traffic.",
        "expected_sources": ["Transport_Layer_Security.txt", "Public-key_cryptography.txt", "Symmetric-key_algorithm.txt", "Diffie-Hellman_key_exchange.txt"]
    },
    {
        "id": "multihop_6",
        "category": "multi-hop",
        "question": "How does reinforcement learning model decision-making using Markov decision processes, and how does Q-learning solve it?",
        "ground_truth": "Reinforcement learning formalizes decision-making using Markov Decision Processes (MDPs), which define states, actions, transition probabilities, and rewards. Q-learning is a model-free RL algorithm that solves this by learning a Q-function that estimates the expected utility of taking an action in a state without knowing the transition probabilities.",
        "expected_sources": ["Reinforcement_learning.txt", "Markov_decision_process.txt", "Q-learning.txt"]
    },
    {
        "id": "multihop_7",
        "category": "multi-hop",
        "question": "What is the relationship between Claude Shannon's work and the foundation of modern digital communication networks?",
        "ground_truth": "Claude Shannon founded information theory, defining the mathematical limits of data compression (entropy) and transmission rate (channel capacity) over noisy channels. Modern computer networks use these principles to design reliable communication protocols (like TCP) that maximize throughput and recover from packet loss.",
        "expected_sources": ["Claude_Shannon.txt", "Information_theory.txt", "Computer_network.txt", "Transmission_Control_Protocol.txt"]
    },
    {
        "id": "multihop_8",
        "category": "multi-hop",
        "question": "How does Git enable collaborative development, and how does CI automate integration of these changes?",
        "ground_truth": "Git allows developers to branch, modify code, and merge changes asynchronously in a local and remote repository. Continuous Integration (CI) systems automatically pull these merged branch changes from Git, trigger unit tests, and compile the code to ensure that changes do not break the main branch.",
        "expected_sources": ["Git.txt", "Continuous_integration.txt", "Unit_testing.txt", "Software_engineering.txt"]
    },
    {
        "id": "multihop_9",
        "category": "multi-hop",
        "question": "How are vector databases used to support Large Language Models in semantic search?",
        "ground_truth": "Large Language Models generate high-dimensional vector embeddings of text that represent semantic meaning. Vector databases store these embeddings and perform fast similarity queries (like cosine similarity) to retrieve relevant context chunks, which are then injected into the LLM's prompt to reduce hallucination.",
        "expected_sources": ["Vector_database.txt", "Large_language_model", "Semantic_search.txt", "Cosine_similarity.txt"]
    },
    {
        "id": "multihop_10",
        "category": "multi-hop",
        "question": "Contrast the execution styles of compilers and interpreters, explaining how a virtual machine can combine them.",
        "ground_truth": "Compilers translate the entire source code into native machine code before execution, resulting in fast execution. Interpreters execute source code line-by-line, which is slower but more flexible. A virtual machine can compile source code to intermediate bytecode, and then interpret or dynamically compile (JIT compile) it to machine code at runtime.",
        "expected_sources": ["Compiler.txt", "Interpreter_computing.txt", "Virtual_machine.txt"]
    },
    {
        "id": "multihop_11",
        "category": "multi-hop",
        "question": "How do decision trees form the foundation of random forests and gradient boosting machines?",
        "ground_truth": "Decision trees are the base learners in ensemble methods. Random forests train multiple independent decision trees in parallel using bootstrap aggregation (bagging) and average their results. Gradient boosting machines train decision trees sequentially, where each new tree is fit to correct the residual errors of the previous trees.",
        "expected_sources": ["Decision_tree.txt", "Random_forest.txt", "Gradient_boosting.txt"]
    },
    {
        "id": "multihop_12",
        "category": "multi-hop",
        "question": "How does a blockchain protect transactions from tampering, and how does Ethereum build smart contracts on top of it?",
        "ground_truth": "A blockchain uses cryptographic hashing to link transactions into blocks, creating a decentralized ledger secured by consensus. Ethereum builds on this by adding a virtual machine (EVM) that executes Turing-complete smart contracts, storing their state and execution results on the immutable ledger.",
        "expected_sources": ["Blockchain.txt", "Ethereum.txt", "Smart_contract.txt", "Cryptography.txt"]
    },
    {
        "id": "multihop_13",
        "category": "multi-hop",
        "question": "How does caching in Redis speed up query times for a relational database like SQL?",
        "ground_truth": "Relational databases (SQL) store data on disk and execute complex queries, which can be slow under load. Redis is an in-memory database that caches the results of these database queries. When the app requests data, it first checks Redis, avoiding slow SQL reads.",
        "expected_sources": ["Redis.txt", "SQL.txt", "Database.txt", "Caching.txt"]
    },
    {
        "id": "multihop_14",
        "category": "multi-hop",
        "question": "What is the difference between overfitting and underfitting, and how can regularization and training data size address them?",
        "ground_truth": "Overfitting is when a model learns the noise in training data, scoring high on training but low on testing. Underfitting is when the model is too simple to capture the patterns. Regularization prevents overfitting by penalizing large model parameters, while increasing training data helps the model generalize.",
        "expected_sources": ["Overfitting.txt", "Regularization_mathematics.txt", "Machine_learning.txt"]
    },
    {
        "id": "multihop_15",
        "category": "multi-hop",
        "question": "Explain how a data pipeline moves raw data to a data warehouse using ETL, and how Apache Spark facilitates this for Big Data.",
        "ground_truth": "A data pipeline extracts raw data from various sources, transforms it (cleansing, sorting, structuring), and loads it (ETL) into a central data warehouse. For big data, Apache Spark distributes these transform tasks across a cluster in memory, speed up processing.",
        "expected_sources": ["Data_pipeline.txt", "Extract,_transform,_load.txt", "Data_warehouse.txt", "Apache_Spark.txt", "Big_data.txt"]
    },
    {
        "id": "multihop_16",
        "category": "multi-hop",
        "question": "How do public keys and private keys function in Diffie-Hellman key exchange vs RSA encryption?",
        "ground_truth": "In RSA, a public key is used to encrypt a message, and only the corresponding private key can decrypt it. In Diffie-Hellman key exchange, both parties exchange public values derived from their private keys to compute a shared symmetric secret key, without actually encrypting data or transmitting the secret key.",
        "expected_sources": ["Public-key_cryptography.txt", "RSA_cryptosystem.txt", "Diffie-Hellman_key_exchange.txt"]
    },
    {
        "id": "multihop_17",
        "category": "multi-hop",
        "question": "What roles do the Controller, Model, and View play in the MVC software pattern, and how does it separate concerns?",
        "ground_truth": "The Model represents the application data and business logic. The View renders the UI presentation. The Controller processes user input, updates the Model, and selects the View. This pattern separates concerns by decoupling data representation from user interaction and display.",
        "expected_sources": ["Model-view-controller.txt", "Software_design_pattern.txt"]
    },
    {
        "id": "multihop_18",
        "category": "multi-hop",
        "question": "How does TCP manage reliability and flow control over IP networks, and why does UDP bypass this?",
        "ground_truth": "TCP manages reliability by assigning sequence numbers to packets, requiring acknowledgements, and retransmitting lost packets. It uses a sliding window mechanism for flow control. UDP bypasses this overhead to enable low-latency, real-time transmissions where packet drops are preferable to delays.",
        "expected_sources": ["Transmission_Control_Protocol.txt", "User_Datagram_Protocol.txt", "Internet_protocol_suite.txt", "Computer_network.txt"]
    },
    {
        "id": "multihop_19",
        "category": "multi-hop",
        "question": "How does clean code practice identify code smells, and how does refactoring resolve technical debt?",
        "ground_truth": "Clean code practices define code quality rules. Code smells are indicators of poorly structured or redundant code. Refactoring is the process of restructuring this internal code without changing external behavior to resolve code smells and pay down technical debt.",
        "expected_sources": ["Clean_code.txt", "Code_smell.txt", "Refactoring.txt", "Technical_debt.txt"]
    },
    {
        "id": "multihop_20",
        "category": "multi-hop",
        "question": "How did the Turing machine concept influence Von Neumann architecture and the modern general-purpose operating system?",
        "ground_truth": "The Universal Turing machine showed that a single machine could execute any program stored as data. This led to the Von Neumann architecture (storing program instructions and data in the same memory). Modern operating systems manage these hardware resources to execute multiple stored programs concurrently.",
        "expected_sources": ["Universal_Turing_machine.txt", "Turing_machine.txt", "Operating_system.txt", "Computer_security.txt"]
    },
    {
        "id": "multihop_21",
        "category": "multi-hop",
        "question": "Compare K-Means clustering and Principal Component Analysis as unsupervised learning techniques.",
        "ground_truth": "K-Means clustering is a partition-based technique that groups n observations into k discrete clusters based on distance to cluster centroids. Principal Component Analysis (PCA) is a dimensionality reduction technique that projects data onto principal components that maximize variance. Both are unsupervised but serve different purposes: clustering vs. projection.",
        "expected_sources": ["K-means_clustering.txt", "Principal_component_analysis.txt", "Unsupervised_learning.txt"]
    },
    {
        "id": "multihop_22",
        "category": "multi-hop",
        "question": "How does semantic search using vector databases differ from traditional lexical search using TF-IDF?",
        "ground_truth": "Lexical search (TF-IDF) matches exact keywords in queries and documents, scoring by term frequencies. Semantic search uses vector databases to query high-dimensional embeddings that represent the meaning of concepts, allowing retrieval of relevant documents even if they share no exact keywords.",
        "expected_sources": ["Tf-idf.txt", "Semantic_search.txt", "Vector_database.txt", "Information_retrieval.txt"]
    },
    {
        "id": "multihop_23",
        "category": "multi-hop",
        "question": "What is a major difference between Docker containers and hypervisor-based virtual machines in terms of OS kernel sharing?",
        "ground_truth": "Virtual machines run on a hypervisor and require a full guest operating system (kernel and all) for each VM, which consumes significant resources. Docker containers share the host operating system's kernel, making them much more lightweight, fast, and resource-efficient.",
        "expected_sources": ["Docker_software.txt", "Virtual_machine.txt", "Operating_system.txt"]
    },
    {
        "id": "multihop_24",
        "category": "multi-hop",
        "question": "How can garbage collection cause latencies in high-performance caching systems, and how can in-memory stores like Redis optimize this?",
        "ground_truth": "Garbage collection (GC) in managed languages can pause application execution ('stop-the-world' pauses) to free memory, which spikes response latency in caching. High-performance systems use C-based memory management (like Redis) or off-heap allocations to manage memory directly and bypass GC pauses.",
        "expected_sources": ["Garbage_collection_computer_science.txt", "Redis.txt", "Caching.txt", "Memcached.txt"]
    },
    {
        "id": "multihop_25",
        "category": "multi-hop",
        "question": "Explain the Chomsky hierarchy and its relation to theoretical automata models like Turing machines.",
        "ground_truth": "The Chomsky hierarchy classifies formal grammars into Regular, Context-Free, Context-Sensitive, and Recursively Enumerable. Each grammar maps to an automaton capable of parsing it: regular grammars to finite state automata, context-free to pushdown automata, and recursively enumerable grammars to Turing machines.",
        "expected_sources": ["Chomsky_hierarchy.txt", "Formal_language.txt", "Automata_theory.txt", "Turing_machine.txt"]
    },

    # === AMBIGUOUS QA (20 pairs) ===
    {
        "id": "ambiguous_1",
        "category": "ambiguous",
        "question": "How do you implement caching in a software application?",
        "ground_truth": "Implementing caching depends on the system tier: you can use client-side browser caching, CDN edge caching, application-level in-memory caches (like local dicts/LRU caches), or distributed caches (like Redis or Memcached). The answer requires specifying the application layer and caching requirements.",
        "expected_sources": ["Caching.txt", "Redis.txt", "Memcached.txt"]
    },
    {
        "id": "ambiguous_2",
        "category": "ambiguous",
        "question": "What is the best database architecture for an application?",
        "ground_truth": "There is no single 'best' database architecture. Relational databases (SQL) are best for structured data and transactions. NoSQL databases (document, key-value) are best for scale and flexibility. Graph databases are best for highly connected data, and vector databases are best for semantic search.",
        "expected_sources": ["Database.txt", "Relational_database.txt", "NoSQL.txt", "Graph_database.txt", "Vector_database.txt"]
    },
    {
        "id": "ambiguous_3",
        "category": "ambiguous",
        "question": "How can I secure my network?",
        "ground_truth": "Securing a network requires a multi-layered security plan: using firewalls and access lists, encrypting data using protocols like TLS, securing communication channels with SSH/VPNs, using strong authentication, and monitoring network traffic. The exact implementation depends on network scale.",
        "expected_sources": ["Computer_security.txt", "Cryptography.txt", "Transport_Layer_Security.txt"]
    },
    {
        "id": "ambiguous_4",
        "category": "ambiguous",
        "question": "What model should I use for natural language processing?",
        "ground_truth": "The choice of model depends on the task and resources: Transformer models (like BERT, GPT, or Gemini) are state-of-the-art for complex language tasks. For simpler tasks like text classification, classic models like Support Vector Machines, Naive Bayes, or TF-IDF with Logistic Regression are more efficient.",
        "expected_sources": ["Transformer_deep_learning_architecture.txt", "Natural_language_processing.txt", "Tf-idf.txt", "Support_vector_machine.txt"]
    },
    {
        "id": "ambiguous_5",
        "category": "ambiguous",
        "question": "How do you design a secure public key system?",
        "ground_truth": "Designing a public-key system requires selecting a secure asymmetric algorithm (like RSA, Diffie-Hellman, or Elliptic Curve Cryptography), managing keys safely, and setting up a Public Key Infrastructure (PKI) with certificates to prevent man-in-the-middle attacks.",
        "expected_sources": ["Public-key_cryptography.txt", "Cryptography.txt", "RSA_cryptosystem.txt", "Diffie-Hellman_key_exchange.txt"]
    },
    {
        "id": "ambiguous_6",
        "category": "ambiguous",
        "question": "What is standard encryption?",
        "ground_truth": "The term 'standard encryption' usually refers to the Advanced Encryption Standard (AES) for symmetric encryption, and RSA or Elliptic Curve Cryptography for asymmetric encryption. In network contexts, it refers to standard protocols like TLS/HTTPS.",
        "expected_sources": ["Advanced_Encryption_Standard.txt", "Public-key_cryptography.txt", "Transport_Layer_Security.txt", "Cryptography.txt"]
    },
    {
        "id": "ambiguous_7",
        "category": "ambiguous",
        "question": "What is the best way to clean up code?",
        "ground_truth": "Cleaning code depends on the specific issues present: it can involve removing dead code, renaming variables for clarity, writing unit tests to cover modifications, and refactoring complex logic into simpler functions to remove code smells and resolve technical debt.",
        "expected_sources": ["Clean_code.txt", "Refactoring.txt", "Code_smell.txt", "Technical_debt.txt"]
    },
    {
        "id": "ambiguous_8",
        "category": "ambiguous",
        "question": "How do you run virtual environments?",
        "ground_truth": "Virtual environments can refer to OS-level virtualization (using virtual machines via hypervisors), containerization (using Docker), or language-specific sandboxing (like Python virtual environments). The answer depends on what you want to isolate.",
        "expected_sources": ["Virtual_machine.txt", "Docker_software.txt", "Operating_system.txt"]
    },
    {
        "id": "ambiguous_9",
        "category": "ambiguous",
        "question": "How should I structure my software engineering project?",
        "ground_truth": "Project structure depends on the architecture style: monolithic vs. microservices, OOP vs. functional programming, or design patterns like MVC. It also depends on using agile methodologies like Scrum and tools like Git for version control.",
        "expected_sources": ["Software_engineering.txt", "Microservices.txt", "Model-view-controller.txt", "Scrum_software_development.txt"]
    },
    {
        "id": "ambiguous_10",
        "category": "ambiguous",
        "question": "How do you scale an application?",
        "ground_truth": "Scaling can be done horizontally (adding more instances of servers or containers using Kubernetes and load balancers) or vertically (adding CPU/RAM to a single machine). It also involves caching (Redis), database optimization, or CDN caching.",
        "expected_sources": ["Load_balancing_computing.txt", "Kubernetes.txt", "Microservices.txt", "Caching.txt"]
    },
    {
        "id": "ambiguous_11",
        "category": "ambiguous",
        "question": "What is attention in machine learning?",
        "ground_truth": "Attention can refer to the general concept of focused computation, but in deep learning, it specifically refers to self-attention mechanisms in Transformers, or alignment models in seq2seq RNNs. It calculates dynamic weighting coefficients for input elements.",
        "expected_sources": ["Attention_machine_learning.txt", "Transformer_deep_learning_architecture.txt", "Recurrent_neural_network.txt"]
    },
    {
        "id": "ambiguous_12",
        "category": "ambiguous",
        "question": "How can I avoid overfitting my model?",
        "ground_truth": "Avoiding overfitting depends on the model type: techniques include adding regularization terms (L1/L2), cross-validation, using simpler models, increasing the size of training data, early stopping, or using ensemble methods like random forests.",
        "expected_sources": ["Overfitting.txt", "Regularization_mathematics.txt", "Random_forest.txt", "Machine_learning.txt"]
    },
    {
        "id": "ambiguous_13",
        "category": "ambiguous",
        "question": "What is SQL query tuning?",
        "ground_truth": "SQL tuning involves optimizing slow queries by adding indexes, rewriting the SQL syntax, avoiding subqueries, or restructuring relational database schemas. The best approach depends on the query planner and data layout.",
        "expected_sources": ["SQL.txt", "Relational_database.txt", "Database.txt"]
    },
    {
        "id": "ambiguous_14",
        "category": "ambiguous",
        "question": "How do you do API development?",
        "ground_truth": "API development involves defining endpoints (REST, gRPC, or GraphQL), handling serialization, implementing authentication/security, and using frameworks (such as FastAPI, Express, or Django). The approach depends on latency and language constraints.",
        "expected_sources": ["Application_programming_interface.txt", "Microservices.txt", "Computer_network.txt"]
    },
    {
        "id": "ambiguous_15",
        "category": "ambiguous",
        "question": "How should I design my database keys?",
        "ground_truth": "In relational databases (SQL), keys refer to primary and foreign keys used for data integrity. In key-value stores (Redis), keys are simple string identifiers. In cryptography, keys refer to public and private key pairs. The term requires clarification.",
        "expected_sources": ["Database.txt", "SQL.txt", "Redis.txt", "Cryptography.txt"]
    },
    {
        "id": "ambiguous_16",
        "category": "ambiguous",
        "question": "What is the best way to handle concurrent users?",
        "ground_truth": "Handling concurrent users requires different techniques at different layers: web servers use load balancing; application servers use async event loops or threading; databases use transaction isolation levels, connection pooling, or distributed caching.",
        "expected_sources": ["Load_balancing_computing.txt", "Database.txt", "Caching.txt", "Computer_network.txt"]
    },
    {
        "id": "ambiguous_17",
        "category": "ambiguous",
        "question": "How do you define an algorithm?",
        "ground_truth": "Informally, an algorithm is a step-by-step procedure for solving a problem. Formally, in computability theory, it is a sequence of states executable by a Turing-complete machine that halts on valid inputs. The level of formality must be specified.",
        "expected_sources": ["Algorithm.txt", "Turing_machine.txt", "Computability_theory.txt"]
    },
    {
        "id": "ambiguous_18",
        "category": "ambiguous",
        "question": "What is public key standard?",
        "ground_truth": "Public key standards can refer to specific cryptographic algorithms (like RSA or Diffie-Hellman), certificate format standards (like X.509), or standard cryptographic protocol suites (like Public-Key Cryptography Standards / PKCS).",
        "expected_sources": ["Public-key_cryptography.txt", "RSA_cryptosystem.txt", "Cryptography.txt"]
    },
    {
        "id": "ambiguous_19",
        "category": "ambiguous",
        "question": "What is agile software development?",
        "ground_truth": "Agile is a set of software development values and principles outlined in the Agile Manifesto. It can be implemented using various frameworks, such as Scrum, Kanban, or Extreme Programming, focusing on iterative delivery.",
        "expected_sources": ["Agile_software_development.txt", "Scrum_software_development.txt", "Software_engineering.txt"]
    },
    {
        "id": "ambiguous_20",
        "category": "ambiguous",
        "question": "What does scaling out mean?",
        "ground_truth": "Scaling out (horizontal scaling) means adding more machine nodes (e.g. servers, containers, database instances) to a system. It is contrasted with scaling up (vertical scaling), which adds resources to a single node. The context (database vs application tier) defines the tooling.",
        "expected_sources": ["Database.txt", "Kubernetes.txt", "Microservices.txt", "Load_balancing_computing.txt"]
    },

    # === UNANSWERABLE QA (25 pairs) ===
    {
        "id": "unanswerable_1",
        "category": "unanswerable",
        "question": "What is the recipe for baking a chocolate chip cookie?",
        "ground_truth": "I am sorry, but the provided text corpus only contains articles on Computer Science, Cryptography, and Machine Learning. It does not contain recipes or cooking instructions.",
        "expected_sources": []
    },
    {
        "id": "unanswerable_2",
        "category": "unanswerable",
        "question": "Who won the FIFA World Cup in 2022?",
        "ground_truth": "I am sorry, but the provided corpus does not contain sports history or FIFA World Cup results.",
        "expected_sources": []
    },
    {
        "id": "unanswerable_3",
        "category": "unanswerable",
        "question": "What is the capital city of Australia?",
        "ground_truth": "I am sorry, but the capital of Australia is not mentioned in the provided technical computer science documents.",
        "expected_sources": []
    },
    {
        "id": "unanswerable_4",
        "category": "unanswerable",
        "question": "How many championships has Michael Jordan won?",
        "ground_truth": "I am sorry, but the corpus contains no information about Michael Jordan or basketball championships.",
        "expected_sources": []
    },
    {
        "id": "unanswerable_5",
        "category": "unanswerable",
        "question": "What was the closing stock price of Apple on May 12, 2026?",
        "ground_truth": "I am sorry, but the technical document corpus does not contain historical stock price data or financial market records.",
        "expected_sources": []
    },
    {
        "id": "unanswerable_6",
        "category": "unanswerable",
        "question": "How do you cure the common cold?",
        "ground_truth": "I am sorry, but there is no medical information or treatment guidelines for the common cold in the computer science corpus.",
        "expected_sources": []
    },
    {
        "id": "unanswerable_7",
        "category": "unanswerable",
        "question": "What is the distance between the Earth and the Moon?",
        "ground_truth": "I am sorry, but astronomical measurements like the distance to the Moon are not present in the provided documents.",
        "expected_sources": []
    },
    {
        "id": "unanswerable_8",
        "category": "unanswerable",
        "question": "Who was the first president of the United States?",
        "ground_truth": "I am sorry, but US presidential history is not covered in the provided technical documents.",
        "expected_sources": []
    },
    {
        "id": "unanswerable_9",
        "category": "unanswerable",
        "question": "How do you play chess?",
        "ground_truth": "I am sorry, but the rules of chess are not documented in the provided corpus.",
        "expected_sources": []
    },
    {
        "id": "unanswerable_10",
        "category": "unanswerable",
        "question": "What is the capital of France?",
        "ground_truth": "I am sorry, but the geography of France is not mentioned in this computer science corpus.",
        "expected_sources": []
    },
    {
        "id": "unanswerable_11",
        "category": "unanswerable",
        "question": "Explain the plot of the movie Inception.",
        "ground_truth": "I am sorry, but movie plots and film reviews are not covered in the provided technical corpus.",
        "expected_sources": []
    },
    {
        "id": "unanswerable_12",
        "category": "unanswerable",
        "question": "What are the symptoms of influenza?",
        "ground_truth": "I am sorry, but the provided corpus has no medical text regarding influenza symptoms.",
        "expected_sources": []
    },
    {
        "id": "unanswerable_13",
        "category": "unanswerable",
        "question": "How do plants perform photosynthesis?",
        "ground_truth": "I am sorry, but plant biology and photosynthesis are not described in the computer science documentation.",
        "expected_sources": []
    },
    {
        "id": "unanswerable_14",
        "category": "unanswerable",
        "question": "Who painted the Mona Lisa?",
        "ground_truth": "I am sorry, but art history and Leonardo da Vinci are not discussed in the provided text files.",
        "expected_sources": []
    },
    {
        "id": "unanswerable_15",
        "category": "unanswerable",
        "question": "What is the height of Mount Everest?",
        "ground_truth": "I am sorry, but geographic parameters like the height of Mount Everest are not included in the database.",
        "expected_sources": []
    },
    {
        "id": "unanswerable_16",
        "category": "unanswerable",
        "question": "What are the ingredients in a Margherita pizza?",
        "ground_truth": "I am sorry, but food recipes are outside the scope of the provided technical documents.",
        "expected_sources": []
    },
    {
        "id": "unanswerable_17",
        "category": "unanswerable",
        "question": "Who won the Academy Award for Best Picture in 2024?",
        "ground_truth": "I am sorry, but the corpus does not track Academy Award winners or entertainment news.",
        "expected_sources": []
    },
    {
        "id": "unanswerable_18",
        "category": "unanswerable",
        "question": "What is the translation of 'Thank you' in Japanese?",
        "ground_truth": "I am sorry, but foreign language translations are not part of the provided technical computer science dataset.",
        "expected_sources": []
    },
    {
        "id": "unanswerable_19",
        "category": "unanswerable",
        "question": "How do you repair a leaky faucet?",
        "ground_truth": "I am sorry, but plumbing instructions are not available in this computer science corpus.",
        "expected_sources": []
    },
    {
        "id": "unanswerable_20",
        "category": "unanswerable",
        "question": "What is the speed of sound in water?",
        "ground_truth": "I am sorry, but physical properties like the speed of sound in water are not contained in the provided documents.",
        "expected_sources": []
    },
    {
        "id": "unanswerable_21",
        "category": "unanswerable",
        "question": "Who is the CEO of Microsoft in 2026?",
        "ground_truth": "I am sorry, but corporate executive history in 2026 is not mentioned in the provided text files.",
        "expected_sources": []
    },
    {
        "id": "unanswerable_22",
        "category": "unanswerable",
        "question": "What is the average lifespan of a domestic cat?",
        "ground_truth": "I am sorry, but veterinary science and information about domestic cats are not present in the corpus.",
        "expected_sources": []
    },
    {
        "id": "unanswerable_23",
        "category": "unanswerable",
        "question": "How do you play the acoustic guitar?",
        "ground_truth": "I am sorry, but music tutorials are outside the scope of this computer science dataset.",
        "expected_sources": []
    },
    {
        "id": "unanswerable_24",
        "category": "unanswerable",
        "question": "What is the rules of American football?",
        "ground_truth": "I am sorry, but sports rules are not covered in the provided technical articles.",
        "expected_sources": []
    },
    {
        "id": "unanswerable_25",
        "category": "unanswerable",
        "question": "What is the distance to the nearest star system, Alpha Centauri?",
        "ground_truth": "I am sorry, but astronomy information regarding Alpha Centauri is not part of this corpus.",
        "expected_sources": []
    }
]

def main():
    print(f"Creating evaluation dataset...")
    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save the 105 QA pairs to file
    with open(EVAL_DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(EVAL_QA_PAIRS, f, indent=2, ensure_ascii=False)
        
    print(f"[+] Successfully wrote {len(EVAL_QA_PAIRS)} QA pairs to {EVAL_DATASET_PATH}")
    
    # Count categories
    counts = {}
    for qa in EVAL_QA_PAIRS:
        cat = qa["category"]
        counts[cat] = counts.get(cat, 0) + 1
    print("Dataset Category Counts:")
    for cat, count in counts.items():
        print(f"  - {cat}: {count}")

if __name__ == "__main__":
    main()
