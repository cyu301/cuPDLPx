/*
Copyright 2025 Haihao Lu

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "cupdlpx.h"
#include "mps_parser.h"

#include <getopt.h>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace
{
const char *termination_reason_to_string(termination_reason_t reason)
{
    switch (reason)
    {
    case TERMINATION_REASON_OPTIMAL:
        return "OPTIMAL";
    case TERMINATION_REASON_PRIMAL_INFEASIBLE:
        return "PRIMAL_INFEASIBLE";
    case TERMINATION_REASON_DUAL_INFEASIBLE:
        return "DUAL_INFEASIBLE";
    case TERMINATION_REASON_TIME_LIMIT:
        return "TIME_LIMIT";
    case TERMINATION_REASON_ITERATION_LIMIT:
        return "ITERATION_LIMIT";
    case TERMINATION_REASON_FEAS_POLISH_SUCCESS:
        return "FEAS_POLISH_SUCCESS";
    case TERMINATION_REASON_UNSPECIFIED:
        return "UNSPECIFIED";
    default:
        return "UNKNOWN";
    }
}

std::string trim(const std::string &input)
{
    size_t start = 0;
    while (start < input.size() &&
           std::isspace(static_cast<unsigned char>(input[start])))
    {
        ++start;
    }
    size_t end = input.size();
    while (end > start &&
           std::isspace(static_cast<unsigned char>(input[end - 1])))
    {
        --end;
    }
    return input.substr(start, end - start);
}

std::string strip_comments(const std::string &input)
{
    size_t hash_pos = input.find('#');
    if (hash_pos == std::string::npos)
    {
        return input;
    }
    return input.substr(0, hash_pos);
}

std::string instance_name_from_path(const std::string &path)
{
    size_t slash = path.find_last_of("/\\");
    std::string base = (slash == std::string::npos) ? path : path.substr(slash + 1);
    size_t dot = base.find('.');
    if (dot != std::string::npos)
    {
        base = base.substr(0, dot);
    }
    return base;
}

std::string escape_csv_field(const std::string &field)
{
    if (field.find_first_of(",\"\n") == std::string::npos)
    {
        return field;
    }

    std::string escaped;
    escaped.reserve(field.size() + 2);
    escaped.push_back('"');
    for (char ch : field)
    {
        if (ch == '"')
        {
            escaped.push_back('"');
            escaped.push_back('"');
        }
        else
        {
            escaped.push_back(ch);
        }
    }
    escaped.push_back('"');
    return escaped;
}

std::vector<std::string> parse_csv_line(const std::string &line)
{
    std::vector<std::string> fields;
    std::string field;
    bool in_quotes = false;

    for (size_t i = 0; i < line.size(); ++i)
    {
        char ch = line[i];
        if (in_quotes)
        {
            if (ch == '"')
            {
                if (i + 1 < line.size() && line[i + 1] == '"')
                {
                    field.push_back('"');
                    ++i;
                }
                else
                {
                    in_quotes = false;
                }
            }
            else
            {
                field.push_back(ch);
            }
        }
        else
        {
            if (ch == '"')
            {
                in_quotes = true;
            }
            else if (ch == ',')
            {
                fields.push_back(field);
                field.clear();
            }
            else
            {
                field.push_back(ch);
            }
        }
    }
    fields.push_back(field);
    return fields;
}

void write_csv_row(std::ostream &out, const std::vector<std::string> &fields)
{
    for (size_t i = 0; i < fields.size(); ++i)
    {
        if (i != 0)
        {
            out << ',';
        }
        out << escape_csv_field(fields[i]);
    }
    out << '\n';
}

std::string to_scientific_string(double value)
{
    std::ostringstream oss;
    oss.setf(std::ios::scientific);
    oss << std::setprecision(17) << value;
    return oss.str();
}

bool is_absolute_path(const std::string &path)
{
    if (path.empty())
    {
        return false;
    }
    if (path[0] == '/' || path[0] == '\\')
    {
        return true;
    }
    if (path.size() > 1 && std::isalpha(static_cast<unsigned char>(path[0])) &&
        path[1] == ':')
    {
        return true;
    }
    return false;
}

std::string join_paths(const std::string &base, const std::string &leaf)
{
    if (base.empty())
    {
        return leaf;
    }
    if (leaf.empty())
    {
        return base;
    }
    char last = base.back();
    if (last == '/' || last == '\\')
    {
        return base + leaf;
    }
    return base + "/" + leaf;
}

std::string directory_of_path(const std::string &path)
{
    size_t slash = path.find_last_of("/\\");
    if (slash == std::string::npos)
    {
        return ".";
    }
    return path.substr(0, slash);
}

bool file_has_content(const std::string &path)
{
    std::ifstream handle(path, std::ios::binary);
    if (!handle)
    {
        return false;
    }
    handle.seekg(0, std::ios::end);
    std::streamoff size = handle.tellg();
    return size > 0;
}

std::unordered_set<std::string> read_existing_datasets(
    const std::string &csv_path)
{
    std::unordered_set<std::string> datasets;
    std::ifstream csv_file(csv_path);
    if (!csv_file)
    {
        return datasets;
    }

    std::string line;
    int dataset_index = -1;
    bool first_row = true;

    while (std::getline(csv_file, line))
    {
        if (line.empty())
        {
            continue;
        }

        std::vector<std::string> fields = parse_csv_line(line);
        if (fields.empty())
        {
            continue;
        }

        if (first_row)
        {
            for (size_t i = 0; i < fields.size(); ++i)
            {
                if (trim(fields[i]) == "dataset")
                {
                    dataset_index = static_cast<int>(i);
                    break;
                }
            }
            first_row = false;
            if (dataset_index >= 0)
            {
                continue;
            }
            dataset_index = 0;
        }

        if (dataset_index >= 0 &&
            dataset_index < static_cast<int>(fields.size()))
        {
            std::string value = trim(fields[dataset_index]);
            if (!value.empty())
            {
                datasets.insert(value);
            }
        }
    }

    return datasets;
}

void print_usage(const char *prog_name)
{
    std::cerr << "Usage: " << prog_name
              << " [OPTIONS] <datasets_txt> <output_csv>\n\n";
    std::cerr << "Options:\n";
    std::cerr << "  -h, --help                          Display this help message.\n";
    std::cerr << "  -v, --verbose                       Enable verbose logging (default: false).\n";
    std::cerr << "  -d, --debug                         Enable debug logging (default: false).\n";
    std::cerr << "      --time_limit <seconds>          Time limit in seconds (default: 3600.0).\n";
    std::cerr << "      --iter_limit <iterations>       Iteration limit (default: 2147483647).\n";
    std::cerr << "      --eps_opt <tolerance>           Relative optimality tolerance (default: 1e-4).\n";
    std::cerr << "      --eps_feas <tolerance>          Relative feasibility tolerance (default: 1e-4).\n";
    std::cerr << "      --eps_infeas_detect <tolerance> Infeasibility detection tolerance (default: 1e-10).\n";
    std::cerr << "      --l_inf_ruiz_iter <int>         Iterations for L-inf Ruiz rescaling (default: 10).\n";
    std::cerr << "      --no_pock_chambolle             Disable Pock-Chambolle rescaling (default: enabled).\n";
    std::cerr << "      --pock_chambolle_alpha <float>  Value for Pock-Chambolle alpha (default: 1.0).\n";
    std::cerr << "      --no_bound_obj_rescaling        Disable bound objective rescaling (default: enabled).\n";
    std::cerr << "      --eval_freq <int>               Termination evaluation frequency (default: 200).\n";
    std::cerr << "      --sv_max_iter <int>             Max iterations for singular value estimation (default: 5000).\n";
    std::cerr << "      --sv_tol <float>                Tolerance for singular value estimation (default: 1e-4).\n";
    std::cerr << "  -f, --feasibility_polishing         Enable feasibility polishing (default: false).\n";
    std::cerr << "      --eps_feas_polish <tolerance>   Relative feasibility polish tolerance (default: 1e-6).\n";
    std::cerr << "\nDatasets file format:\n";
    std::cerr << "  First non-empty line: dataset root directory\n";
    std::cerr << "  Subsequent lines: dataset paths relative to the root (or absolute paths)\n";
}
} // namespace

int main(int argc, char *argv[])
{
    pdhg_parameters_t params;
    set_default_parameters(&params);

    static struct option long_options[] = {
        {"help", no_argument, 0, 'h'},
        {"verbose", no_argument, 0, 'v'},
        {"debug", no_argument, 0, 'd'},
        {"time_limit", required_argument, 0, 1001},
        {"iter_limit", required_argument, 0, 1002},
        {"eps_opt", required_argument, 0, 1003},
        {"eps_feas", required_argument, 0, 1004},
        {"eps_infeas_detect", required_argument, 0, 1005},
        {"eps_feas_polish", required_argument, 0, 1006},
        {"feasibility_polishing", no_argument, 0, 'f'},
        {"l_inf_ruiz_iter", required_argument, 0, 1007},
        {"pock_chambolle_alpha", required_argument, 0, 1008},
        {"no_pock_chambolle", no_argument, 0, 1009},
        {"no_bound_obj_rescaling", no_argument, 0, 1010},
        {"sv_max_iter", required_argument, 0, 1011},
        {"sv_tol", required_argument, 0, 1012},
        {"eval_freq", required_argument, 0, 1013},
        {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "hvfd", long_options, nullptr)) != -1)
    {
        switch (opt)
        {
        case 'h':
            print_usage(argv[0]);
            return 0;
        case 'v':
            params.verbose = true;
            break;
        case 'd':
            params.debug = true;
            break;
        case 1001: // --time_limit
            params.termination_criteria.time_sec_limit = atof(optarg);
            break;
        case 1002: // --iter_limit
            params.termination_criteria.iteration_limit = atoi(optarg);
            break;
        case 1003: // --eps_opt
            params.termination_criteria.eps_optimal_relative = atof(optarg);
            break;
        case 1004: // --eps_feas
            params.termination_criteria.eps_feasible_relative = atof(optarg);
            break;
        case 1005: // --eps_infeas_detect
            params.termination_criteria.eps_infeasible = atof(optarg);
            break;
        case 1006: // --eps_feas_polish
            params.termination_criteria.eps_feas_polish_relative = atof(optarg);
            break;
        case 'f': // --feasibility_polishing
            params.feasibility_polishing = true;
            break;
        case 1007: // --l_inf_ruiz_iter
            params.l_inf_ruiz_iterations = atoi(optarg);
            break;
        case 1008: // --pock_chambolle_alpha
            params.pock_chambolle_alpha = atof(optarg);
            break;
        case 1009: // --no_pock_chambolle
            params.has_pock_chambolle_alpha = false;
            break;
        case 1010: // --no_bound_obj_rescaling
            params.bound_objective_rescaling = false;
            break;
        case 1011: // --sv_max_iter
            params.sv_max_iter = atoi(optarg);
            break;
        case 1012: // --sv_tol
            params.sv_tol = atof(optarg);
            break;
        case 1013: // --eval_freq
            params.termination_evaluation_frequency = atoi(optarg);
            break;
        case '?':
            return 1;
        }
    }

    if (argc - optind != 2)
    {
        std::cerr << "Error: You must specify a datasets file and an output CSV.\n\n";
        print_usage(argv[0]);
        return 1;
    }

    const std::string datasets_path = argv[optind];
    const std::string csv_path = argv[optind + 1];

    std::ifstream datasets_file(datasets_path);
    if (!datasets_file)
    {
        std::cerr << "Failed to open datasets file: " << datasets_path << '\n';
        return 1;
    }

    std::unordered_set<std::string> processed_datasets =
        read_existing_datasets(csv_path);
    bool csv_has_content = file_has_content(csv_path);

    std::ofstream csv_file;
    if (csv_has_content)
    {
        csv_file.open(csv_path, std::ios::app);
    }
    else
    {
        csv_file.open(csv_path);
    }
    if (!csv_file)
    {
        std::cerr << "Failed to open CSV output file: " << csv_path << '\n';
        return 1;
    }

    const std::vector<std::string> header = {
        "dataset",
        "instance",
        "termination_reason",
        "runtime_sec",
        "iterations_count",
        "primal_objective_value",
        "dual_objective_value",
        "relative_primal_residual",
        "relative_dual_residual",
        "absolute_objective_gap",
        "relative_objective_gap",
        "feasibility_polishing_time_sec",
        "feasibility_polishing_iteration_count"};

    if (!csv_has_content)
    {
        write_csv_row(csv_file, header);
    }

    std::string line;
    int line_number = 0;
    int solved_count = 0;
    int failed_count = 0;
    int skipped_count = 0;
    std::string dataset_root;
    const std::string datasets_dir = directory_of_path(datasets_path);

    while (std::getline(datasets_file, line))
    {
        ++line_number;
        std::string cleaned = trim(strip_comments(line));
        if (cleaned.empty())
        {
            continue;
        }

        if (dataset_root.empty())
        {
            dataset_root = cleaned;
            if (!is_absolute_path(dataset_root))
            {
                dataset_root = join_paths(datasets_dir, dataset_root);
            }
            continue;
        }

        std::string dataset_path = cleaned;
        if (!is_absolute_path(dataset_path))
        {
            dataset_path = join_paths(dataset_root, dataset_path);
        }

        if (processed_datasets.find(dataset_path) != processed_datasets.end())
        {
            ++skipped_count;
            continue;
        }
        processed_datasets.insert(dataset_path);

        const std::string instance = instance_name_from_path(dataset_path);
        std::vector<std::string> row(header.size(), "");
        row[0] = dataset_path;
        row[1] = instance;

        lp_problem_t *problem = read_mps_file(dataset_path.c_str());
        if (problem == nullptr)
        {
            std::cerr << "Failed to read MPS file at line " << line_number
                      << ": " << dataset_path << '\n';
            row[2] = "READ_ERROR";
            write_csv_row(csv_file, row);
            csv_file.flush();
            ++failed_count;
            continue;
        }

        cupdlpx_result_t *result = solve_lp_problem(problem, &params);
        if (result == nullptr)
        {
            std::cerr << "Solver failed for dataset at line " << line_number
                      << ": " << dataset_path << '\n';
            row[2] = "SOLVER_ERROR";
            write_csv_row(csv_file, row);
            csv_file.flush();
            lp_problem_free(problem);
            ++failed_count;
            continue;
        }

        row[2] = termination_reason_to_string(result->termination_reason);
        row[3] = to_scientific_string(result->cumulative_time_sec);
        row[4] = std::to_string(result->total_count);
        row[5] = to_scientific_string(result->primal_objective_value);
        row[6] = to_scientific_string(result->dual_objective_value);
        row[7] = to_scientific_string(result->relative_primal_residual);
        row[8] = to_scientific_string(result->relative_dual_residual);
        row[9] = to_scientific_string(result->objective_gap);
        row[10] = to_scientific_string(result->relative_objective_gap);
        if (result->feasibility_polishing_time > 0.0)
        {
            row[11] = to_scientific_string(result->feasibility_polishing_time);
            row[12] = std::to_string(result->feasibility_iteration);
        }

        write_csv_row(csv_file, row);
        csv_file.flush();

        cupdlpx_result_free(result);
        lp_problem_free(problem);
        ++solved_count;
    }

    if (dataset_root.empty())
    {
        std::cerr << "Datasets file does not define a dataset root path.\n";
        return 1;
    }

    std::cerr << "Batch complete: " << solved_count << " solved, "
              << failed_count << " failed, " << skipped_count << " skipped.\n";

    return failed_count == 0 ? 0 : 2;
}
