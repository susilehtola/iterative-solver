#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_LOGGER_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_LOGGER_H
#include <array>
#include <format>
#include <iomanip>
#include <iterator>
#include <limits>
#include <ranges>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <typeinfo>
#include <tuple>
#include <utility>

#include <molpro/iostream.h>


namespace molpro::linalg::itsolv {
  class Logger;
}

namespace molpro::linalg::itsolv::log {

/*!
* @brief Different levels of logging
*/
enum class Verbosity : short {
  Trace,
  Debug,
  Info,
  None,
};


/*!
* @brief Severity of a given message
*/
enum class Severity : short {
  Normal,
  Warning,
  Error,
  Fatal,
};


/*!
 * Concept to check for a nested static variable 'name' whose type is convertible to const char *
 */
template <typename T>
concept name_tagged = requires {
  { T::name } -> std::convertible_to<const char *>;
};


/*!
 * Base class for all context classes, which serve as a compile-time identifiable log contextss.
 */
template< typename CRTP, bool fixed_args, typename ...Args >
struct ContextBase {
  /*!
   * The name of this logging context
   */
  static constexpr const char *name = []() {
    if constexpr (name_tagged<CRTP>) {
      return CRTP::name;
    }
    return typeid(CRTP).name();
  };

  /*!
   * Whether calls using this log context use a fixed set of arguments
   */
  static constexpr bool uses_fixed_arguments = fixed_args;

  // Used only as a store for the arg type list
  using arg_types_tuple = std::tuple<std::remove_cvref_t<Args>...>;

  /*!
   * Number of arguments calls using this log context will use (if using a fixed argument set)
   */
  static constexpr std::size_t num_args = sizeof...(Args);

  /*!
   * Converts the given object to the type it should be handled as in downstream log handlers.
   * For non-ranges that means ensuring the type is const a const reference whereas for ranges,
   * we convert them to a view (as const ranges can't be iterated over in the general case).
   */
  template<typename T>
  static constexpr decltype(auto) to_arg(T &&arg) {
    if constexpr (std::ranges::view<T>) {
      return std::forward<T>(arg);
    } else if constexpr (std::ranges::range<std::remove_cvref_t<T>>) {
      return std::ranges::views::all(arg);
    } else {
      return std::as_const(arg);
    }
  }

  /*!
   * The type if the Idx-th argument that uses of this log context imply. Only applies if the context
   * actually uses a fixed argument set.
   */
  template<std::size_t Idx>
  requires(Idx < num_args && uses_fixed_arguments)
  using arg_t = decltype(to_arg(std::declval<std::tuple_element_t<Idx, arg_types_tuple>>()));

  /*!
   * Extracts the requested argument from the provided argument pack
   */
  template<std::size_t Idx, typename ...Ts>
  static arg_t<Idx> get_arg(Ts &&...args) {
    return to_arg(std::get<Idx>(std::forward_as_tuple(std::forward<Ts>(args)...)));
  }
};


/*!
 * Concept checking whether a given type is considered a Context.
 */
template< typename T >
concept context = requires(const T &t) {
  // If this lambda can be called, T must inherit from ContextBase (with arbitrary template parameter)
  []<bool b, typename...Ts>(const ContextBase<T, b, Ts...> &){}(t);
};

/*!
 * Concept checking that the provided arguments match with the ones specified in the logging context
 */
template< typename Ctx, typename ...Args >
concept context_uses_correct_arg_types = context<Ctx> && std::derived_from<Ctx, ContextBase<Ctx, Ctx::uses_fixed_arguments, Args...>>;

/*!
 * Concept checking that a given type is pair-like object
 */
template <typename T>
concept pair_like = requires(const std::remove_cvref_t<T> &t) {
  t.first;
  t.second;
};

/*!
 * Concept checking that a given type supports operator<< into an output stream
 */
template<typename T>
concept streamable = requires (std::ostream &stream, const std::remove_cvref_t<T> &t) {
  stream << t;
};

/*!
 * Concept checking that a given type can be formatted via std::format
 */
template<typename T>
concept formattable = requires (const std::remove_cvref_t<T> &t, std::format_context ctx) {
  std::formatter<std::remove_cvref_t<T>>().format(t, ctx);
};

/*!
 * Concept checking that the provided range can be converted into a string by either
 * using std::format or operator<<
 */
template <typename T>
concept string_convertible_range =
    std::ranges::range<T> && (formattable<std::ranges::range_value_t<T>> || streamable<std::ranges::range_value_t<T>> ||
                              pair_like<std::ranges::range_value_t<T>>);

/*!
 * Compile-time, constexpr-capable (fixed-size) string-like object
 */
template<std::size_t MaxSize>
requires(MaxSize > 0)
class ConstexprString {
public:
  constexpr ConstexprString() : m_data(), m_size(0) {
    m_data.fill('\0');
  }
  template<std::size_t N>
  requires(N - 1 <= MaxSize)
  constexpr ConstexprString(const char (&literal)[N]) : ConstexprString() {
    // -1 as we don't want to copy the terminating null character
    for (std::size_t i = 0; i < N - 1; ++i) {
      push_back(literal[i]);
    }
  }
  constexpr ~ConstexprString() {}

  constexpr void push_back(char c) {
    if (m_size == MaxSize) {
      throw std::out_of_range("Maximum capacity has been reached - can't append further chars");
    }

    m_data.at(m_size) = c;
    ++m_size;
  }

  constexpr std::size_t size() const { return m_size; }

  constexpr char &operator[](std::size_t idx) {
    if (idx >= m_size) {
      throw std::out_of_range("Requested index is out of range");
    }

    return m_data.at(idx);
  }

  constexpr std::string_view as_view() const {
    return {m_data.data(), m_size};
  }

  constexpr operator std::string_view() const {
    return as_view();
  }

private:
  std::array<char, MaxSize> m_data;
  std::size_t m_size = 0;
};


/*!
 * @returns Number of digits required to represent the given value in base 10
 */
constexpr std::size_t num_digits(std::size_t val) {
  if (val == 0) {
    return 1;
  }

  std::size_t num = 0;

  while (val > 0){
    val /= 10;
    ++num;
  }

  return num;
}


/*!
 * Creates a format string that contains instruction to use the given precision while formatting.
 * The selling point is that this is suitable to be used in a constexpr/consteval context.
 */
template<std::size_t precision>
constexpr ConstexprString<num_digits(precision) + 4> create_fmt_string_for_precision() {
  ConstexprString<num_digits(precision) + 4> str;

  str.push_back('{');
  str.push_back(':');
  str.push_back('.');

  if (precision == 0) {
    str.push_back('0');
  } else {
    std::size_t begin = str.size();

    // This create the digits in reverse order
    std::size_t prec_val = precision;
    while (prec_val > 0) {
      std::size_t tmp = prec_val / 10;
      std::size_t digit = prec_val - tmp * 10;
      prec_val = tmp;

      str.push_back('0' + static_cast<char>(digit));
    }

    std::size_t num_digits = str.size() - begin;

    // Reverse order of digits in-place
    for (std::size_t i = 0; i < num_digits / 2; ++i) {
      std::swap(str[i + begin], str[str.size() - i - 1]);
    }
  }

  str.push_back('}');

  return str;
}


/*!
* Constant indicating that formatting should be done with default precision
*/
constexpr const std::size_t default_precision = std::numeric_limits<std::size_t>::max();


/*!
 * Customization point for controlling how exactly a type is formatted
 *
 * @tparam Context The context in which the formatting is happening
 * @tparam T The type to format
 * @tparam precision The requested precision (default_precision indicates unspecified/default precision)
 */
template<context Context, typename T, std::size_t precision = default_precision>
struct FormatOption {
  /*!
   * Yields the format string to be used with std::format
   */
  static constexpr auto format_string() {
    if constexpr (precision != default_precision) {
      return create_fmt_string_for_precision<precision>();
    } else {
      return ConstexprString<2>{"{}"};
    }
  }
  /*!
   * @returns String to be inserted at the beginning of a range
   */
  static constexpr std::string_view range_begin_mark() { return "["; }
  /*!
   * @returns String to be inserted at the end of a range
   */
  static constexpr std::string_view range_end_mark() { return "]"; }
  /*!
   * @returns Separator that shall be used when formatting a range of T
   */
  static constexpr std::string_view range_separator() { return ", "; }
  /*!
   * Sets the stream's properties when using operator<< to yield a string representation
   */
  static constexpr void prepare_stream(std::ostream &stream) {
    if constexpr (precision != default_precision) {
      stream << std::setprecision(precision);
    }
  }
};

/*!
 * Converts the given type into a string
 */
template <context Context, std::size_t precision = default_precision, typename T>
  requires(formattable<T> || streamable<T> || string_convertible_range<T> || pair_like<T>)
static std::string stringify(T &&t) {
  using FOpts = FormatOption<Context, std::remove_cvref_t<T>, precision>;

  if constexpr (formattable<T>) {
    static constexpr auto fmt = FOpts::format_string();
    return std::format(fmt.as_view(), t);
  } else if constexpr (streamable<T>) {
    std::stringstream sstream;
    FOpts::prepare_stream(sstream);
    sstream << t;
    return sstream.str();
  } else if constexpr (pair_like<T>) {
    std::stringstream sstream;

    sstream << "{ " << stringify<Context, precision>(t.first) << ": " << stringify<Context, precision>(t.second)
            << " }";

    return sstream.str();
  } else {
    std::stringstream sstream;

    using std::ranges::begin;
    using std::ranges::end;

    sstream << FOpts::range_begin_mark();

    for (auto it = begin(t); it != end(t); ++it) {
      sstream << stringify<Context, precision>(*it);

      if (it + 1 != end(t)) {
        sstream << FOpts::range_separator();
      }
    }

    sstream << FOpts::range_end_mark();

    return sstream.str();
  }
}


/*!
 * Concept checking that the stringify function can be called on this
 * context-precision-type combination
 */
template<typename Context, std::size_t precision, typename T>
concept stringify_supported = requires (std::remove_cvref_t<T> &t) {
  { stringify<Context, precision>(t) } -> std::same_as<std::string>;
};



// Declaring some default logging contexts

/*!
 * Default context for logging - used if we don't have a more specific context
 */
struct Generic : ContextBase<Generic, false> { static constexpr const char *name = "Generic"; };
static_assert(context<Generic>);
/*!
 * Context used when dumping data
 */
struct DataDump : ContextBase<DataDump, false> { static constexpr const char *name = "DataDump"; };
static_assert(context<DataDump>);



/*!
 * Customization point for logging in case one needs direct access to the data instead of
 * its stringified form. If a suitable template specialization exists, this will be called
 * instead of the default handler in the logger object (for which the data is previously
 * converted into string format).
 */
template< context Context >
struct LogHandler {
  template<typename ...Ts>
  static void handle(const Logger &logger, Severity severity, Verbosity verbosity, std::string_view message, Ts...args) = delete;

  template<typename ...Ts>
  void handle(Severity severity, Verbosity verbosity, std::string_view message, Ts...args) = delete;
};

template< typename Context, typename ...Args >
concept static_handler_exists = requires (const Logger &logger, Args ...args) {
  LogHandler<Context>::handle(logger, std::declval<Severity>(), std::declval<Verbosity>(),
      std::declval<std::string_view>(), args...);
};
template< typename Context, typename ...Args >
concept member_handler_exists = requires (const LogHandler<Context> &handler, Args ...args) {
  handler.handle(std::declval<Severity>(), std::declval<Verbosity>(),
      std::declval<std::string_view>(), args...);
};

/*!
 * Concept checking whether there exists a suitable specialization of the LogHandler class
 */
template< typename Context, typename ...Args >
concept handler_exists = static_handler_exists<Context, Args...> || member_handler_exists<Context, Args...>;


} // namespace molpro::linalg::itsolv::log



namespace molpro::linalg::itsolv {

/*!
 * Logger class implementation. Provides (customizable) logging functionalities.
 */
class Logger {
public:
  Logger(log::Severity min_severity = log::Severity::Normal, log::Verbosity verbosity = log::Verbosity::Info, bool enable_data_dumps = false)
    : m_min_severity(min_severity), m_verbosity(verbosity), m_dump_data(enable_data_dumps) {}

  /*!
   * General logging function - generally, you will want to use one of the convenience wrappers around this function
   */
  template< log::context Context = log::Generic, std::size_t precision = log::default_precision, typename ...Ts >
  void msg(log::Severity severity, log::Verbosity verbosity, std::string_view message, Ts && ...args) const {
    log_ctx(Context::name, sizeof...(args));

    if (verbosity < m_verbosity || severity < m_min_severity) {
      return;
    }

    static_assert((log::stringify_supported<Context, precision, Ts> && ...), "All types passed to logger must be stringify-able");
    if constexpr (Context::uses_fixed_arguments) {
      static_assert(log::context_uses_correct_arg_types<Context, std::remove_cvref_t<Ts>...>, "Log function called with parameters that are inconsistent with associated context");
    }

    if constexpr (log::handler_exists<Context, Ts...>) {
      using Handler = log::LogHandler<Context>;
      if constexpr (log::static_handler_exists<Context, Ts...> && log::member_handler_exists<Context, Ts...>) {
        // Need to figure out whether to call static or member function
        const Handler *handler = dynamic_cast<const Handler *>(this);
        if (handler) {
          handler->handle(severity, verbosity, message, std::forward<Ts>(args)...);
        } else {
          Handler::handle(*this, severity, verbosity, message, std::forward<Ts>(args)...);
        }
      } else if constexpr (log::member_handler_exists<Context, Ts...>) {
        dynamic_cast<const Handler &>(*this).handle(severity, verbosity, message, std::forward<Ts>(args)...);
      } else {
        static_assert(log::static_handler_exists<Context, Ts...>);
        Handler::handle(*this, severity, verbosity, message, std::forward<Ts>(args)...);
      }
    } else if constexpr (sizeof...(args) > 0) {
      std::string arg_str = ((log::stringify<Context, precision>(std::forward<Ts>(args)) + ", ") + ...);

      if (!message.empty() && message.back() != ' ') {
        // If message doesn't end with a blank, we assume simply concatenating the argument string
        // to it would produce a nonsense message. Hopefully a generic separator will do better.
        arg_str = "  ->  " + arg_str;
      }

      default_message_handler(Context::name, severity, verbosity, std::string(message) + arg_str);
    } else {
      default_message_handler(Context::name, severity, verbosity, message);
    }
  }

  template< log::context Context = log::Generic, typename ...Ts>
  void trace(std::string_view message, Ts && ...args) const {
    msg<Context>(log::Severity::Normal, log::Verbosity::Trace, message, std::forward<Ts>(args)...);
  }

  template< log::context Context = log::Generic, typename ...Ts>
  void debug(std::string_view message, Ts && ...args) const {
    msg<Context>(log::Severity::Normal, log::Verbosity::Debug, message, std::forward<Ts>(args)...);
  }

  template< log::context Context = log::Generic, typename ...Ts>
  void info(std::string_view message, Ts && ...args) const {
    msg<Context>(log::Severity::Normal, log::Verbosity::Info, message, std::forward<Ts>(args)...);
  }

  template< log::context Context = log::Generic, typename ...Ts>
  void warn(std::string_view message, Ts && ...args) const {
    msg<Context>(log::Severity::Warning, log::Verbosity::None, message, std::forward<Ts>(args)...);
  }

  template< log::context Context = log::Generic, typename ...Ts>
  void error(std::string_view message, Ts && ...args) const {
    msg<Context>(log::Severity::Error, log::Verbosity::None, message, std::forward<Ts>(args)...);
  }

  template< log::context Context = log::Generic, typename ...Ts>
  void fatal(std::string_view message, Ts && ...args) const {
    msg<Context>(log::Severity::Fatal, log::Verbosity::None, message, std::forward<Ts>(args)...);
  }


  template<std::size_t precision = log::default_precision, typename ...Ts>
  void data_dump(std::string_view what, Ts...data) const {
    if (!m_dump_data) {
      log_ctx(log::DataDump::name, sizeof...(data));
      return;
    }

    msg<log::DataDump, precision>(log::Severity::Normal, log::Verbosity::Trace, what, data...);
  }

  log::Severity min_severity() const { return  m_min_severity; }

  void set_min_severity(log::Severity severity) { m_min_severity = severity; }

  log::Verbosity verbosity() const { return  m_verbosity; }

  void set_verbosity(log::Verbosity verbosity) { m_verbosity = verbosity; }

  bool data_dumps_enabled() const { return m_dump_data; }

  void enable_data_dumps(bool enable) { m_dump_data = enable; }

protected:
  // These functions are the traditional customization points by means of subclassing and overwriting

  virtual void default_message_handler(std::string_view ctx, log::Severity severity, log::Verbosity verbosity, std::string_view msg) const {
    bool add_colon = false;
    switch (severity) {
    case log::Severity::Normal:
        break;
    case log::Severity::Warning:
        molpro::cout << "[WARNING]";
        add_colon = true;
        break;
    case log::Severity::Error:
        molpro::cout << "[ERROR]:";
        add_colon = true;
        break;
    case log::Severity::Fatal:
        molpro::cout << "[FATAL]:";
        add_colon = true;
        break;
    }
    switch (verbosity) {
    case log::Verbosity::None:
        break;
    case log::Verbosity::Info:
        molpro::cout << "[INFO]";
        add_colon = true;
        break;
    case log::Verbosity::Debug:
        molpro::cout << "[DEBUG]";
        add_colon = true;
        break;
    case log::Verbosity::Trace:
        molpro::cout << "[TRACE]";
        add_colon = true;
        break;
    };

    if (ctx != log::Generic::name) {
      molpro::cout << "[" << ctx << "]";
      add_colon = true;
    }

    molpro::cout << (add_colon ? ": " : "") << msg;
    if (severity >= log::Severity::Error) {
      molpro::cout << std::endl;
    } else {
      molpro::cout << "\n";
    }
  }

  virtual void log_ctx(std::string_view ctx, std::size_t num_args) const {
    // By default we don't do anything
  }

private:
  /// Highest verbosity that will still be logged
  log::Verbosity m_verbosity = log::Verbosity::Info;
  /// Highest severity that will still be logged
  log::Severity m_min_severity = log::Severity::Normal;
  /// Whether data dumps shall be performed
  bool m_dump_data = false;
};

} // namespace molpro::linalg::itsolv

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_LOGGER_H
