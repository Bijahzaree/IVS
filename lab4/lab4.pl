% Плотников Алексей Васильевич
% ЭВМбз-22-1
% (Prolog) Разработка экспертной системы (предметная область выбирается самостоятельно)


% ==================================================================
% Экспертная система: диагностика Ethernet (L1/L2/L3)
% С поддержкой VLAN, STP, PortFast, BPDUguard и объяснений
% ==================================================================

:- dynamic answer/3.
:- dynamic reason/1.

clear_answers :-
    retractall(answer(_, _, _)).
clear_reasons :-
    retractall(reason(_)).

ask(Question, Allowed) :-
    ( answer(Question, Value, Allowed) -> true
    ; format('~w ~w: ', [Question, Allowed]),
      read(Reply),
      ( member(Reply, Allowed) ->
          true
      ; format('Недопустимый ответ. Допустимо: ~w~n', [Allowed]),
        fail
      ),
      assert(answer(Question, Reply, Allowed)),
      assert(reason(question(Question, Reply)))
    ).

ask_yesno(Question) :-
    ask(Question, [yes, no]).

get_answer(Question, Value) :-
    answer(Question, Value, _).

% ---------- L1 ----------
diagnose_physical(repair(Text)) :-
    ask_yesno("Горит ли Link (индикатор на обоих концах)?"),
    ( get_answer("Горит ли Link (индикатор на обоих концах)?", no) ->
        Text = "Нет физического соединения. Проверьте кабель, SFP, порт. Убедитесь в совпадении скорости/дуплекса (auto-negotiation)."
    ; ask_yesno("Наблюдаются ли ошибки на интерфейсе (CRC, фреймы)?") ->
        ( get_answer("Наблюдаются ли ошибки на интерфейсе (CRC, фреймы)?", yes) ->
            Text = "Обнаружены ошибки на физическом уровне. Замените кабель, проверьте согласование duplex/speed (ethtool / show interface)."
        ; fail
        )
    ; fail
    ).

% ---------- L2 ----------
diagnose_layer2(repair(Text)) :-
    ask_yesno("MAC-адрес устройства изучен в таблице коммутации?"),
    ( get_answer("MAC-адрес устройства изучен в таблице коммутации?", no) ->
        Text = "MAC-адрес не изучен. Проверьте: порт в правильном VLAN, trunk разрешён, нет фильтрации MAC, порт не заблокирован STP."
    ; check_vlan_and_stp(Text)
    ).

check_vlan_and_stp(Text) :-
    ask("Порт находится в режиме access или trunk?", [access, trunk]),
    get_answer("Порт находится в режиме access или trunk?", Mode),
    ( Mode == trunk ->
        ask_yesno("В trunk разрешён VLAN вашего устройства?"),
        ( get_answer("В trunk разрешён VLAN вашего устройства?", no) ->
            Text = "Ваш VLAN не разрешён в trunk. Добавьте VLAN в список разрешённых (switchport trunk allowed vlan)."
        ; check_stp(Text)
        )
    ; Mode == access ->
        ask_yesno("PVID порта совпадает с VLAN вашего устройства?"),
        ( get_answer("PVID порта совпадает с VLAN вашего устройства?", no) ->
            Text = "Несоответствие PVID. Настройте access-порт в правильный VLAN (switchport access vlan)."
        ; check_stp(Text)
        )
    ).

% Изменённый check_stp с вопросами про PortFast и BPDUguard
check_stp(Text) :-
    ask_yesno("Состояние порта по STP – Forwarding?"),
    ( get_answer("Состояние порта по STP – Forwarding?", no) ->
        % Порт заблокирован
        ask_yesno("Включён ли PortFast на этом порту?"),
        ( get_answer("Включён ли PortFast на этом порту?", yes) ->
            ask_yesno("Порт переходит в состояние err-disable (BPDUguard сработал)?"),
            ( get_answer("Порт переходит в состояние err-disable (BPDUguard сработал)?", yes) ->
                Text = "Порт переведён в err-disable из-за BPDUguard. Это может быть вызвано подключением коммутатора к порту с PortFast или петлей в сети. Отключите PortFast на транковых портах или настройте BPDUguard только на edge-портах."
            ; Text = "Порт заблокирован STP, хотя PortFast включён. Проверьте, не приходят ли BPDU на этот порт (например, подключено другое коммутирующее устройство). Для портов, подключенных к конечным устройствам, убедитесь, что PortFast настроен правильно (spanning-tree portfast)."
            )
        ; Text = "Порт заблокирован STP (состояние BLOCKING/LISTENING). Для порта, подключённого к конечному устройству (рабочая станция, сервер), рекомендуется включить PortFast (spanning-tree portfast). Проверьте корневой мост и отсутствие петель."
        )
    ; ask_yesno("Наблюдается ли MAC flapping (миграция MAC между портами)?") ->
        ( get_answer("Наблюдается ли MAC flapping (миграция MAC между портами)?", yes) ->
            ask_yesno("Включён ли BPDUguard или loopguard на проблемных портах?"),
            ( get_answer("Включён ли BPDUguard или loopguard на проблемных портах?", yes) ->
                Text = "MAC flapping может быть вызван петлёй. Проверьте конфигурацию STP, возможно, BPDUguard перевёл порт в err-disable, но затем он восстановился. Убедитесь, что loopguard включён для портов с корневыми мостами."
            ; Text = "MAC flapping указывает на петлю. Включите STP (если не включён), проверьте настройки BPDUguard и loopguard. Настройте portfast только на edge-портах."
            )
        ; fail
        )
    ; fail
    ).

% ---------- L3 ----------
diagnose_layer3(repair(Text)) :-
    ask_yesno("Есть ли ARP-запись на шлюзе для IP устройства?"),
    ( get_answer("Есть ли ARP-запись на шлюзе для IP устройства?", no) ->
        Text = "ARP-запись отсутствует. Проверьте: IP-адрес и маску, нет ли конфликта IP, устройство отвечает на ARP."
    ; ask_yesno("Пингуется ли шлюз по умолчанию?") ->
        ( get_answer("Пингуется ли шлюз по умолчанию?", no) ->
            Text = "Шлюз не отвечает. Проверьте default gateway, ACL, файрвол, включена ли маршрутизация."
        ; ask_yesno("Пингуется ли внешний адрес (8.8.8.8)?") ->
            ( get_answer("Пингуется ли внешний адрес (8.8.8.8)?", no) ->
                Text = "Шлюз доступен, но внешний адрес не пингуется. Проблема у провайдера, маршрутизации на шлюзе или блокировка ICMP."
            ; Text = "Сетевые соединения работают. Проблема на прикладном уровне (DNS, firewall, приложение)."
            )
        ; fail
        )
    ; fail
    ).

record_rule(RuleName) :-
    assert(reason(rule(RuleName))).

diagnose_with_explain(Repair) :-
    clear_reasons,
    ( diagnose_physical(Repair) -> record_rule(physical_layer)
    ; diagnose_layer2(Repair) -> record_rule(layer2_vlan_stp)
    ; diagnose_layer3(Repair) -> record_rule(layer3_arp_routing)
    ; Repair = repair("Не удалось определить причину. Проверьте логи вручную."),
      record_rule(fallback)
    ),
    explain.

explain :-
    findall(Step, reason(Step), Steps),
    writeln("\n=== ОБЪЯСНЕНИЕ ДИАГНОЗА ==="),
    writeln("На основе следующих ответов:"),
    forall(member(question(Q, A), Steps),
           format("- ~w : ~w~n", [Q, A])),
    writeln("\nЦепочка рассуждений:"),
    forall(member(rule(R), Steps), format("- ~w~n", [R])).

run :-
    clear_answers,
    writeln("\n=== ДИАГНОСТИКА ETHERNET (L1/L2/L3) с объяснением ==="),
    writeln("Отвечайте вариантами из списка.\n"),
    diagnose_with_explain(repair(Message)),
    writeln("\n*** РЕКОМЕНДАЦИЯ ***"),
    writeln(Message),
    writeln("").

:- initialization(run).